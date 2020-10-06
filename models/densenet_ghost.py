'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=True),
            # nn.BatchNorm1d(init_channels),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=True),
            # nn.BatchNorm1d(new_channels),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:]



class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, stride=1):
        super(Bottleneck, self).__init__()

        # self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_planes),
            nn.ReLU(True),
            # nn.Conv1d(in_planes, 4*growth_rate, kernel_size=1, stride=stride, padding=1,bias=False),
            # depthwise_conv(in_planes, 4*growth_rate,1),
            nn.Conv1d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, stride=1,
                      padding=1,
                      groups=in_planes
                      ),
            nn.Dropout(0.5)
        )

        # self.bn2 = nn.BatchNorm1d(4*growth_rate)
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(in_planes),
            nn.ReLU(True),
            # nn.Conv1d(4*growth_rate, growth_rate, kernel_size=3, stride=stride, bias=False),
            # depthwise_conv(4*growth_rate,growth_rate),
            nn.Conv1d(
                in_channels=in_planes,
                out_channels=growth_rate,
                kernel_size=1,
                stride=1,
                # padding=0,
                groups=1
            ),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        # self.bn = nn.BatchNorm1d(in_planes)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_planes),
            nn.ReLU(True),
            nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        out = self.conv(x)
        # out = F.avg_pool1d(out, 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=2):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Sequential(
            GhostModule(1, num_planes, kernel_size=7, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm1d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.adaptive_avg_pool1d(F.relu(self.bn(out)), 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,6,6,6], growth_rate=12)

def test():
    from torchsummary import summary
    net = densenet_cifar()
    summary(net.cuda(), (1, 6000))
    # x = torch.randn(1,1,32)
    # y = net(x)
    # print(y)

if __name__ == "__main__":
    test()
