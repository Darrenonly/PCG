'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x



class MyBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, stride=1):
        super(MyBlock, self).__init__()
        self.stride = stride
        mid_planes = int(growth_rate / 4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_planes),
            # nn.ReLU(True),
            h_swish(),
            nn.Conv2d(mid_planes, mid_planes, 3, stride, 1, bias=False, groups=mid_planes),
            nn.BatchNorm2d(mid_planes),
            nn.Conv2d(mid_planes, growth_rate, 1, 1, 0, bias=False),
            nn.BatchNorm2d(growth_rate),
            h_swish(),
            # nn.ReLU(True),
            nn.Dropout(0.3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, 3, stride,1, bias=False, groups=mid_planes),
            nn.BatchNorm2d(mid_planes),
            # nn.ReLU(True),
            nn.Conv2d(mid_planes, growth_rate, 1, 1, 0, bias=False),
            nn.BatchNorm2d(growth_rate),
            h_swish(),
            # nn.ReLU(True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        # out1 = torch.cat([out1, x], 1)
        if self.stride == 2:
            out2 = self.conv2(x)
            out1 = torch.cat([out1, out2], 2)
            # out1 = torch.cat([out1, x], 1)

        return out1

class Channel_Attention(nn.Module):

    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d(1)
        self.__max_pool = nn.AdaptiveMaxPool2d(1)

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel//r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()


    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1+y2)
        return x * y


class Spartial_Attention(nn.Module):

    def __init__(self, kernel_size):
        super(Spartial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()


        self.conv = nn.Sequential(
            BasicConv2d(in_planes, out_planes, activation=True, kernel_size=1),
            # nn.AvgPool1d(kernel_size=3, stride=1),
        )
        self.ca = Channel_Attention(out_planes, 4)
        self.sa = Spartial_Attention(3)
        # self.fc1 = nn.Conv1d(out_planes, out_planes // 4, kernel_size=1)
        # self.fc2 = nn.Conv1d(out_planes // 4, out_planes, kernel_size=1)
    def forward(self, x):
        out = self.conv(x)
        # Squeeze
        # w = F.avg_pool1d(out, out.size(2))
        out = self.ca(out) * out
        # out = self.sa(out) * out

        # w = F.relu(self.fc1(out))
        # w = torch.sigmoid(self.fc2(w))
        # # Excitation
        # out = out * w
        # out = F.avg_pool1d(out, 1)
        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class _StemBlock(nn.Module):
    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features/2)

        self.stem1 = BasicConv2d(num_input_channels, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem2a = BasicConv2d(num_init_features, num_stem_features, kernel_size=1, stride=1)
        self.stem2b = BasicConv2d(num_stem_features, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem3 = BasicConv2d(2*num_init_features, num_init_features, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], 1)
        out = self.stem3(out)

        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, nstride, growth_rate=12, reduction=0.5, num_classes=2):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Sequential(
            _StemBlock(3, num_planes),
            # nn.Conv1d(1, num_planes, kernel_size=7, stride=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], nstride[0])
        num_planes += nblocks[0]*growth_rate
        # self.concat1 = nn.Sequential(nn.Conv1d(num_planes, ))
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], nstride[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], nstride[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3],nstride[3])
        # num_planes += nblocks[3]*growth_rate
        # out_planes = int(math.floor(num_planes * reduction))
        # self.trans4 = Transition(num_planes, out_planes)
        # num_planes = out_planes

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)




    def _make_dense_layers(self, block, in_planes, nblock, nstride):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, nstride))
            in_planes = self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        output = []
        output.append(out)
        for dense1 in self.dense1:
            out = dense1(out)
            output.append(out)
        out = torch.cat(output, dim=1)
        out = self.trans1(out)
        output = []
        output.append(out)
        for dense2 in self.dense2:
            out = dense2(out)
            output.append(out)
        out = torch.cat(output, dim=1)
        out = self.trans2(out)
        output = []
        output.append(out)
        for dense3 in self.dense3:
            out = dense3(out)
            output.append(out)
        out = torch.cat(output, dim=1)
        out = self.trans3(out)
        # output = []
        # output.append(out)
        # for dense4 in self.dense4:
        #     out = dense4(out)
        #     output.append(out)
        # out = torch.cat(output, dim=1)
        # out = self.trans4(out)
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1)
        out1 = out.view(out.size(0), -1)
        out = self.linear(out1)

        # out = F.log_softmax(out, dim=1)
        return out, out1


def densenet_cifar():
    return DenseNet(MyBlock, [4,4,4], [1,1,1],growth_rate=12)

def test():
    from torchsummary import summary
    net = densenet_cifar()
    # print(net)
    summary(net.cuda(), (3, 256, 256))
    # x = torch.randn(1,1,32)
    # y = net(x)
    # print(y)

if __name__ == "__main__":
    test()
