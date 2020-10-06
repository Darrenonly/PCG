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
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import visdom
import torchvision
from config.config import *
from train_test.data_loader import read_feature_from_csv
from models.autoencoder import AutoEncoder
from models.GhostNet import ghost_net
from models.mydnn2 import MyDnn
# from demo.models.googlenet import GoogLeNet
from train_test.my_dataset1 import My_Dataset
from utils.utils import train


def train1():
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((224, 224)),
        # transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((224, 224)),
        # transforms.ToTensor(),
    ])

    train_data = My_Dataset(filepath=FEATPATH + 'train1.txt', transform=train_transforms)
    test_data = My_Dataset(filepath=FEATPATH + 'validation.txt', transform=test_transforms)
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False, num_workers=1)
    print('num_of_trainData:', len(train_data))
    print('num_of_testData:', len(test_data))

    origin_time = time()
    model = AutoEncoder()

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
    criterion = nn.MSELoss().cuda()

    if torch.cuda.is_available():
        net = model.cuda()
    prev_time = datetime.now()
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        # print(train_data)
        for im, label in train_data:
            # print(np.shape(im))
            # writer.add_image('img_origin', im.item(), 20)
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                # label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                # label = Variable(label)
            # forward
            im = im.view(-1, 1, 6000)
            im_y = im
            encoded, decoded = net(im)

            loss = criterion(decoded, im_y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            # train_acc += get_acc(output, label)
            # print("epoch: %d, tain_acc: %f" %(epoch, get_acc(output, label)))
            # writer.add_scalar("train_loss", train_loss, 50)
            # writer.add_scalar("train_acc", train_acc, 50)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if test_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in test_data:

                if torch.cuda.is_available():
                    im = Variable(im.cuda(), volatile=True)
                    # label = Variable(label.cuda(), volatile=True)
                else:
                    im = Variable(im, volatile=True)
                    # label = Variable(label, volatile=True)
                im = im.view(-1, 1, 6000)
                im_y = im
                encoded, decoded = net(im)
                loss = criterion(decoded, im_y)
                # valid_loss+=float(loss.data[0])
                valid_loss += loss.item()
                # valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(test_data),
                       valid_acc / len(test_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        # writer.add_scalar()
        print(epoch_str + time_str)
    torch.save(net.state_dict(), "../checkpoint/autoencoder.pth")


if __name__ == "__main__":
    train1()
