"""
构建心音深度神经网络提取心音特征

输入为 3 x 224 x 224 的心音时频图

网络设计结构
@Input
    CNN(inp=3,oup=16,stride=2,kernel_size=3)
    BN
    RELU(inplace=True)

@layer-n
    [inverted resblock(ghostbottleneck)] x n
    [SElayer]

@Output
    fc x 2
    linear
    softmax



"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class MyDnn(nn.Module):
    def __init__(self, num_classes=2):
        super(MyDnn,self).__init__()

        # first layer
        self.input = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        # middle layer
        self.middle = nn.Sequential(
            Block(32, 64, 2, 2, start_with_relu=False, grow_first=True),
            Block(64, 128, 2, 2, start_with_relu=True, grow_first=True),
            # Block(256, 512, 2, 2, start_with_relu=True, grow_first=True),

            # Block(128, 128, 3, 1, start_with_relu=True, grow_first=True),
            # SELayer(128),
            # Block(128, 128, 3, 1, start_with_relu=True, grow_first=True),
            # # SELayer(128),
            # Block(128, 128, 2, 2, start_with_relu=True, grow_first=True),
            # # SELayer(128),
            # Block(128, 128, 3, 1, start_with_relu=True, grow_first=True),
            # SELayer(128),
            # Block(128, 256, 3, 2, start_with_relu=True, grow_first=False),
        )

        # last layer
        self.last = nn.Sequential(
            SeparableConv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            SeparableConv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),

        )

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.input(x)
        x = self.middle(x)
        x = self.last(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

def test():
    from torchsummary import summary
    net = MyDnn()
    # summary(net.cuda(), (3,224,224))
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
