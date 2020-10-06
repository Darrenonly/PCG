from datetime import datetime

import torch
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from torch import nn
from torch.autograd import Variable
import numpy as np
import random

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from train_test.pytorchtool import EarlyStopping
from sklearn.model_selection import KFold


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())




def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(net, train_data, num_epochs, optimizer, criterion):
    kfold = KFold(n_splits=10)
    min_loss = 100000  # 随便设置一个比较大的数

    for k, (train_index, val_index) in enumerate(kfold.split(train_data)):
        print("=============第"  + str(k+1) + "折开始===============")
        train_db1 = torch.utils.data.Subset(train_data,train_index)
        val_db = torch.utils.data.Subset(train_data,val_index)
        train_loader = DataLoaderX(dataset=train_db1, batch_size=128, shuffle=True, num_workers=2)
        val_loader = DataLoaderX(dataset=val_db, batch_size=128, shuffle=True, num_workers=1)

        print('num_of_trainData:', len(train_db1))
        print('num_of_testData:', len(train_loader))
        print('num_of_valData:', len(val_db))
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        if torch.cuda.is_available():
            net = net.cuda()
            # model = autoencoder.cuda()
        early_stopping = EarlyStopping(patience=20, verbose=True)

        prev_time = datetime.now()
        # model.eval()
        for epoch in range(num_epochs):
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
                output,out1 = net(im)
                # writer.add_image('img_out_' + label, output, 20)
                # print(label)
                loss = criterion(output, label)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

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

                    if torch.cuda.is_available():
                        im = Variable(im.cuda(), volatile=True)
                        label = Variable(label.cuda(), volatile=True)
                    else:
                        im = Variable(im, volatile=True)
                        label = Variable(label, volatile=True)
                    # im = im.view(-1, 1, 6000)
                    # encoded, decoded = model(im)
                    output,out1 = net(im)
                    loss = criterion(output, label)
                    # valid_loss+=float(loss.data[0])
                    valid_loss += loss.item()
                    valid_acc += get_acc(output, label)
                val_loss = valid_loss / len(val_loader)
                if val_loss < min_loss:
                    min_loss = val_loss
                    print("save model")
                    torch.save(net.state_dict(), '../checkpoint/chk6/best1.pth')
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


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(
        in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x + out, True)


class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2), residual_block(64, 64), residual_block(64, 64))

        self.block3 = nn.Sequential(
            residual_block(64, 128, False), residual_block(128, 128))

        self.block4 = nn.Sequential(
            residual_block(128, 256, False), residual_block(256, 256))

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512), nn.AvgPool2d(3))

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
