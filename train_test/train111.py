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
import visdom
import torchvision
from config.config import *
# from train_test.data_loader import read_feature_from_csv
# from models.mobilenet import MobleNetV1
# from models.GhostNet import ghost_net
# from models.mydnn2 import MyDnn
# from demo.models.googlenet import GoogLeNet
from train_test.eval import eval_net
from train_test.my_dataset1 import My_Dataset
from utils.utils import train, get_acc
from models.autoencoder import AutoEncoder
from models.densenet_stem import densenet_cifar
from models.resnet import ResNet18
from torch.autograd import Variable

from sklearn.model_selection import KFold


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def train1():
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((224, 224)),
        # transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((224, 224)),
        # transforms.ToTensor(),
    ])
    kfold = KFold(n_splits=5, shuffle=True)

    train_data = My_Dataset(filepath=FEATPATH + 'train1.txt', transform=train_transforms)
    # train_db, test_db = torch.utils.data.random_split(train_data, [34542, 1818])
    # train_db, test_db = torch.utils.data.random_split(train_data, [2985, 157])
    #
    # test_loader = DataLoaderX(dataset=test_db, batch_size=128, shuffle=False, num_workers=1)
    model = densenet_cifar()

    # test_data = My_Dataset(filepath= FEATPATH + 'validation.txt', transform=test_transforms)
    # print(kfold.split(train_data))

    # 3s
    train_db, val_db = torch.utils.data.random_split(train_data, [32724, 3636])
    val1_db, test_db = torch.utils.data.random_split(val_db, [1818, 1818])
    # 1s
    # train_db, val_db = torch.utils.data.random_split(train_data, [73800, 8199])
    # val1_db, test_db = torch.utils.data.random_split(val_db, [4099, 4100])
    # 2s
    # train_db, val_db = torch.utils.data.random_split(train_data, [63775, 7086])
    # val1_db, test_db = torch.utils.data.random_split(val_db, [3543, 3543])
    # 4s
    # train_db, val_db = torch.utils.data.random_split(train_data, [22326, 2481])
    # val1_db, test_db = torch.utils.data.random_split(val_db, [1241, 1240])
    # 5s
    # train_db, val_db = torch.utils.data.random_split(train_data, [16933, 1881])
    # val1_db, test_db = torch.utils.data.random_split(val_db, [941, 940])
    # train_db, val_db = torch.utils.data.random_split(train_data, [2828, 314])
    # val1_db, test_db = torch.utils.data.random_split(val_db, [157, 157])
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoaderX(dataset=train_db, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoaderX(dataset=val1_db, batch_size=64, shuffle=True, num_workers=1)
    test_loader = DataLoaderX(dataset=test_db, batch_size=64, shuffle=False, num_workers=1)

    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    # train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=1)
    # test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False, num_workers=1)
    print('num_of_trainData:', len(train_db))
    print('num_of_testData:', len(test_db))
    print('num_of_valData:', len(val1_db))

    # # 显示一个batch
    # viz = visdom.Visdom(env='train_hs')
    # viz.image(torchvision.utils.make_grid(next(iter(train_loader))[0], nrow=8), win='train-image')

    # train_label, train_data = read_feature_from_csv(type='train')
    # validation_label, validation_data = read_feature_from_csv(type='validation')
    # # print(train_data)
    # b = train_data[0]
    # print(b.shape)
    # print(np.shape(train_data))
    # input_shape = (None, b.shape[1], b.shape[2])
    # origin_time = time()
    # model = densenet_cifar()
    # logging.info(model.summary())
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([2.0, 1.0])).float()).cuda()
    min_loss = 100000  # 随便设置一个比较大的数
    min_acc = 0.001
    # train(model, train_db, 20, optimizer, criterion)
    #
    if torch.cuda.is_available():
        net = model.cuda()

    prev_time = datetime.now()
    # model.eval()
    for epoch in range(150):
        train_loss = 0
        train_acc = 0
        net.train()
        # print(train_data)
        for im, label in train_loader:
            # print(np.shape(im))
            # writer.add_image('img_origin', im.item(), 20)
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            # im = im.view(-1, 1, 6000)
            # encoded, decoded = model(im)
            output, out1 = net(im)
            # writer.add_image('img_out_' + label, output, 20)
            # print(label)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            train_acc += get_acc(output, label)

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
                # im = im.view(-1, 1, 6000)
                # encoded, decoded = model(im)
                output, out1 = net(im)
                loss = criterion(output, label)
                # valid_loss+=float(loss.data[0])
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            val_loss = valid_loss / len(val_loader)
            val_acc = valid_acc / len(val_loader)
            # if val_loss < min_loss:
            #     min_loss = val_loss
            #     print("save model")
            #     torch.save(net.state_dict(), '../checkpoint/chk7/best1.pth')
            if val_acc > min_acc:
                min_acc = val_acc
                print("save model")
                torch.save(net.state_dict(), '../checkpoint/chk11'
                                             '/best1.pth')
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_loader),
                       train_acc / len(train_loader), valid_loss / len(val_loader),
                       valid_acc / len(val_loader)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_loader),
                          train_acc / len(train_loader)))
        prev_time = cur_time
        # writer.add_scalar()
        print(epoch_str + time_str)

    # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([2,1])).float()).cuda()
    # train(model, train_db, 20, opitmizer, criterion)
    torch.save(model.state_dict(), "../checkpoint/model5.pth")

    return test_loader, len(test_db)


if __name__ == "__main__":
    # loader, num = train1()
    # train_data = My_Dataset(filepath= FEATPATH + 'train1.txt', transform=None)
    # train_loader = DataLoaderX(dataset=train_data, batch_size=128, shuffle=False, num_workers=2)
    train_data = My_Dataset(filepath=FEATPATH + 'train1.txt', transform=None)

    train_db, val_db = torch.utils.data.random_split(train_data, [30865, 3636])
    val1_db, test_db = torch.utils.data.random_split(val_db, [1818, 1818])
    test_loader = DataLoaderX(dataset=test_db, batch_size=64, shuffle=True, num_workers=1)

    model = densenet_cifar()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('../checkpoint\chk1\\best.pth'))
    eval_net(test_loader=test_loader, datanum=len(test_db), net=model)
