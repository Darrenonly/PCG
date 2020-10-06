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
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv1d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm1d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv1d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm1d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv1d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm1d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv1d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm1d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool1d(3, strides, 1))
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


class Block1(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block1, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv1d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(group_width)
        self.conv2 = nn.Conv1d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(group_width)
        self.conv3 = nn.Conv1d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MyDnn(nn.Module):
    def __init__(self, num_classes=2):
        super(MyDnn, self).__init__()

        # first layer
        self.input = nn.Sequential(
            nn.Conv1d(16, 24, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(True),
            nn.Conv1d(24, 24, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(True),
        )

        # middle layer
        self.middle = nn.Sequential(
            nn.Conv1d(24, 48, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(True),
            nn.Conv1d(48, 64, stride=1, kernel_size=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 96, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(True),
            nn.Conv1d(96, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Conv1d(128, 256, stride=2, kernel_size=5, padding=1),
            # nn.BatchNorm1d(256),
            # nn.ReLU(True),
            # nn.Conv1d(256, 512, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(True),
            # Block(24, 96, 3, 1, start_with_relu=False, grow_first=True),
            # Block(96, 128, 3, 1, start_with_relu=True, grow_first=True),
            # # # Block(256, 512, 2, 2, start_with_relu=True, grow_first=True),
            # #
            # Block(128, 128, 3, 2, start_with_relu=True, grow_first=True),
            # # SELayer(128),
            # Block(128, 128, 3, 1, start_with_relu=True, grow_first=True),
            # # SELayer(128),
            # Block(128, 256, 3, 2, start_with_relu=True, grow_first=True),
            # # SELayer(128),
            # Block(256, 256, 3, 1, start_with_relu=True, grow_first=True),
            # # SELayer(128),
            # Block(128, 256, 3, 2, start_with_relu=True, grow_first=False),
            # Block1(24, 4, 32),
            # nn.Dropout(0.5),
            # Block1(2 * 4 * 32, 4, 64),
            # nn.Dropout(0.75),
            # Block1(2 * 4 * 64, 4, 128),
            # nn.Dropout(0.5),
            # Block1(2 * 4 * 128, 4, 256),
            # nn.Dropout(0.5),
            # Block1(2 * 4 * 256, 4, 512),
        )

        # last layer
        # self.last = nn.Sequential(
        #     SeparableConv1d(128, 256, 3, 1, 1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(True),
        #     SeparableConv1d(256, 512, 3, 2, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(True)
        # )

        self.classifier = nn.Sequential(
            # nn.Linear(128, 96),
            # nn.AdaptiveAvgPool1d(3, 1),
            nn.Linear(128, num_classes)

        )

        # ------- init weights --------
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.input(x)
        x = self.middle(x)
        # x = self.last(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


def test():
    from torchsummary import summary
    net = MyDnn()
    # summary(net.cuda(), (3,224,224))
    x = torch.randn(1, 1, 1000)
    y = net(x)
    print(y)


if __name__ == "__main__":
    test()
