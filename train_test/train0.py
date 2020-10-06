"""
@suthor: Hulk
"""
from datetime import datetime

import numpy as np
from time import time
import logging
import os
import random

import torch
from prefetch_generator import BackgroundGenerator
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

import torchvision
from config.config import *

from train_test.eval import eval_net
from train_test.my_dataset1 import My_Dataset
from utils.utils import train, get_acc

from models.densenet_stem import densenet_cifar
# from models.densenet_without_any import densenet_cifar_without_any

from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from train_test.CB import FocalLoss

bs = 128


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight





def train1(train_loader, val_loader, test_loader, model=None, output="checkpoint/chk"):

    if not os.path.exists(output):
        os.mkdir(output)

    # train_data = My_Dataset(filepath= FEATPATH + 'train1.txt', transform=None)
    #
    # train_db, val_db = torch.utils.data.random_split(train_data, [28177, 3130])
    # val1_db, test_db = torch.utils.data.random_split(val_db, [1565, 1565])

    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    # train_loader = DataLoaderX(dataset=train_db, batch_size=64, shuffle=True, num_workers=0)
    # val_loader = DataLoaderX(dataset=val1_db, batch_size=64, shuffle=True, num_workers=0)
    # test_loader = DataLoaderX(dataset=test_db, batch_size=64, shuffle=False, num_workers=0)

    print('num_of_trainData:', len(train_loader)*bs)
    print('num_of_testData:', len(test_loader)*bs)
    print('num_of_valData:', len(val_loader)*bs)

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0002)
    # criterion = FocalLoss(2, alpha=0.7, gamma=1, size_average=True).cuda()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 4])).float()).cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    min_loss = 100000  # 随便设置一个比较大的数2

    min_acc = 0.001

    if torch.cuda.is_available():
        net = model.cuda()


    prev_time = datetime.now()
    # model.eval()
    with open(os.path.join(output,"train5.txt"), 'a', encoding="utf-8") as f:
        for epoch in range(150):

            train_loss = 0
            train_acc = 0
            net.train()

            for im, label in train_loader:
                if torch.cuda.is_available():
                    im = Variable(im.cuda())  # (bs, 3, h, w)
                    label = Variable(label.cuda())  # (bs, h, w)
                else:
                    im = Variable(im)
                    label = Variable(label)
                optimizer.zero_grad()

                out, out1 = net(im)
                # loss = CB_loss(label,out,[4, 1], 2, "softmax", 0.9999, 2.0)
                loss = criterion(out, label)
                # tb.add_graph(model=net, input_to_model=im)
                loss.backward()
                optimizer.step()


                train_loss += loss.data.item()
                train_acc += get_acc(out, label)

            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            if val_loader is not None:
                valid_loss = 0
                valid_acc = 0
                net.eval()
                for im, label in val_loader:
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            im = Variable(im.cuda())
                            label = Variable(label.cuda())
                        else:
                            im = Variable(im)
                            label = Variable(label)

                    out, out1 = net(im)
                    loss = criterion(out, label)
                    # loss = CB_loss(label, out, [4, 1], 2, "softmax", 0.9999, 2.0)

                    valid_loss += loss.item()
                    valid_acc += get_acc(out, label)
                    # valid_acc += get_acc(out, label)
                val_loss = valid_loss / len(val_loader)
                val_acc = valid_acc / len(val_loader)
                # if val_loss < min_loss:                #     min_loss = val_loss
                #     print("save model")
                #     torch.save(net.state_dict(), '../checkpoint/chk7/best1.pth')
                if val_acc > min_acc:
                    min_acc = val_acc
                    print("save model")
                    torch.save(net.state_dict(), os.path.join(output, 'best5.pth'))
                epoch_str = (
                        "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f,"
                        % (epoch, train_loss / len(train_loader),
                           train_acc / len(train_loader), valid_loss / len(val_loader),
                           valid_acc / len(val_loader)))
            else:
                epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                             (epoch, train_loss / len(train_loader),
                              train_acc / len(train_loader)))
            prev_time = cur_time
            # writer.add_scalar()
            tb.add_scalar("train_loss", train_loss / len(train_loader), epoch)
            tb.add_scalar("valid_loss", valid_loss / len(val_loader), epoch)
            tb.add_scalar("train_acc", train_acc / len(train_loader), epoch)
            tb.add_scalar("valid_acc", valid_acc / len(val_loader), epoch)

            print(epoch_str + time_str)
            f.write(epoch_str + time_str + "\n")

        # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([2,1])).float()).cuda()
        # train(model, train_db, 20, opitmizer, criterion)
    # torch.save(model.state_dict(), "../checkpoint/model5.pth")

    return test_loader, len(test_loader)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    # 设置随机数种子
    setup_seed(42)

    train_data = My_Dataset(filepath=FEATPATH + 'train5.txt', transform=None)
    val_data = My_Dataset(filepath=FEATPATH + "validation5.txt", transform=None)
    test_data = My_Dataset(filepath=FEATPATH + "test5.txt", transform=None)
    # tensorboard
    tb = SummaryWriter()
    # x = round(len(train_data)*0.8)
    # y = len(train_data) - x
    # train_db, val_db = torch.utils.data.random_split(train_data, [x, y])
    # val1_db, test_db = torch.utils.data.random_split(val_db, [round(y/2), y-round(y/2)])
    from torchsampler import ImbalancedDatasetSampler
    # For unbalanced dataset we create a weighted sampler
    # weights = make_weights_for_balanced_classes(train_data.imgs, 2)
    # weights = torch.DoubleTensor(weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    sampler = ImbalancedDatasetSampler(train_data)
    # train_loader = DataLoaderX(dataset=train_data, batch_size=bs, shuffle=False, sampler=sampler, num_workers=2)
    train_loader = DataLoaderX(dataset=train_data, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = DataLoaderX(dataset=val_data, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoaderX(dataset=test_data, batch_size=bs, shuffle=False, num_workers=0)
    # model1 = densenet_cifar_without_any()
    # if torch.cuda.is_available():
    #     model1 = model1.cuda()
    # loader1, num1 = train1(train_loader,val_loader,test_loader,model=model1,output="../checkpoint/chk11")
    # model1.load_state_dict(torch.load('../checkpoint/chk11/best1.pth'))
    # eval_net(test_loader=loader1, datanum=num1, net=model1)
    #
    # model2 = densenet_cifar_normal()
    # if torch.cuda.is_available():
    #     model2 = model2.cuda()
    # loader2, num2 = train1(train_loader,val_loader,test_loader, model=model2, output="../checkpoint/chk12")
    # model2.load_state_dict(torch.load('../checkpoint/chk12/best1.pth'))
    # eval_net(test_loader=loader2, datanum=num2, net=model2)

    model3 = densenet_cifar()
    if torch.cuda.is_available():
        model3 = model3.cuda()
    loader3, num3 = train1(train_loader,val_loader,test_loader, model=model3, output="../checkpoint/chk16")
    model3.load_state_dict(torch.load('../checkpoint/chk16/best5.pth'))
    eval_net(test_loader=loader3, datanum=num3*bs, net=model3)

    # file = "../feat_sr1k_0/training-a/a0058/a0058_0.npy"
    # img = np.load(file)
    # img = torch.from_numpy(img)
    # img = img.type(torch.FloatTensor)  # 转Float
    # img = torch.reshape(img, (-1, 1, len(img)))
    # starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    # for _ in range(10):
    #     _ = model(img.cuda())
    # model.eval()
    # with torch.no_grad():
    #
    #     starter.record()
    #     output, out1 = model(img.cuda())
    #     ender.record()
    #     torch.cuda.synchronize()
    #     cur_time = starter.elapsed_time(ender)
    #
    #     print(cur_time)
    tb.close()
    exit(0)