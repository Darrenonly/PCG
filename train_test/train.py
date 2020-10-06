"""
@suthor: Hulk
"""
from datetime import datetime
import numpy as np
import os
import torch
from prefetch_generator import BackgroundGenerator
from torch import nn
from torch.utils.data import DataLoader
from config.config import *
from train_test.eval import eval_net
from train_test.my_dataset import My_Dataset
from utils.utils import train, get_acc
from models.densenet_stem import densenet_cifar
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


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
    # 检测输出路径是否存在
    if not os.path.exists(output):
        os.mkdir(output)
    print('num_of_trainData:', len(train_loader) * bs)
    print('num_of_testData:', len(test_loader) * bs)
    print('num_of_valData:', len(val_loader) * bs)

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.001)        # 参数优化
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([3, 1])).float()).cuda()
    min_acc = 0.001

    if torch.cuda.is_available():
        net = model.cuda()

    prev_time = datetime.now()
    # model.eval()
    with open(os.path.join(output, "train.txt"), 'a', encoding="utf-8") as f:
        for epoch in range(1, 151):
            train_loss = 0
            train_acc = 0
            net.train()
            if epoch % 10 == 0:
                lr = optimizer.param_groups[0]["lr"] * 0.99
                optimizer.param_groups[0]["lr"] = lr
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
                loss = criterion(out, label.long())
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
                    loss = criterion(out, label.long())
                    # loss = CB_loss(label, out, [4, 1], 2, "softmax", 0.9999, 2.0)

                    valid_loss += loss.item()
                    valid_acc += get_acc(out, label)
                val_loss = valid_loss / len(val_loader)
                val_acc = valid_acc / len(val_loader)
                if val_acc > min_acc:
                    min_acc = val_acc
                    print("save model")
                    torch.save(net.state_dict(), os.path.join(output, 'best1.pth'))
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
    return test_loader, len(test_loader)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":

    # 设置随机数种子
    setup_seed(42)
    train_data = My_Dataset(filepath=FEATPATH + 'train5.npy', transform=None)
    val_data = My_Dataset(filepath=FEATPATH + "validation5.npy", transform=None)
    test_data = My_Dataset(filepath=FEATPATH + "test5.npy", transform=None)
    # tensorboard
    tb = SummaryWriter()
    train_loader = DataLoaderX(dataset=train_data, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = DataLoaderX(dataset=val_data, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoaderX(dataset=test_data, batch_size=bs, shuffle=False, num_workers=0)
    model3 = densenet_cifar()
    if torch.cuda.is_available():
        model3 = model3.cuda()
    loader3, num3 = train1(train_loader, val_loader, test_loader, model=model3, output="../checkpoint/chk15")
    model3.load_state_dict(torch.load('../checkpoint/chk15/best1.pth'))
    eval_net(test_loader=loader3, datanum=num3 * bs, net=model3)
    tb.close()
    exit(0)
