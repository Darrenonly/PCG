"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
"""
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from cnn.cnn_utils import CBR, CDilated, CB, BR

__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )


class EESP(nn.Module):
    '''
    EESP类定义了两个函数，初始化函数和前向传播，前向传播按照
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    进行运算
    '''

    def __init__(self, nIn, nOut, stride=1, k=1, r_lim=7, down_method='esp'):  # down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels 输入通道数
        :param nOut: number of output channels 输出通道数
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2 步长
        :param k: # of parallel branches 并行卷积的分支个数
        :param r_lim: A maximum value of receptive field allowed for EESP block EESP模块的最大感受野
        :param g: number of groups to be used in the feature map reduction step. 分组卷积的参数
        '''
        super().__init__()
        self.stride = stride  # 初始化步长
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n,
                                                                                                n1)  # 分支深度卷积中，膨胀率最大时的维度要和输出维度相同

        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)  # 初始化2D卷积，然后归一化,再用PRELU去线性化

        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}  # 对3*3卷积核，膨胀率和膨胀卷积核大小之间的对应关系
        self.k_sizes = list()
        for i in range(k):  # 膨胀率和膨胀卷积核大小之间的对应关系
            ksize = int(3 + 2 * i)
            # 到达边界后
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)

        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):  # 初始化膨胀卷积函数
            # Transform
            d_rate = map_receptive_ksize[self.k_sizes[i]]  # 每轮的膨胀率
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))  # 将所有分支的膨胀函数装到spp_dw中

        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)  # 卷积操作后归一化
        self.br_after_cat = BR(nOut)  # 规范化，BR函数为PRELU和归一化
        self.module_act = nn.PReLU(nOut)  # 去线性化
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        "前向传播算法"

        # Reduce，将M维输入降维到D=N/K维
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # 计算每个分支的输出并依次融合
        # Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)  # 每个分支进行DDConv
            # HFF,从最小的膨胀卷积核的输出开始，逐级叠加，为了改善网格效应
            out_k = out_k + output[k - 1]
            # 保存下来每个分支的结果再融合
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp(  # 先将输出拼接
            self.br_after_cat(  # 然后规范化
                torch.cat(output, 1)  # 使用1*1的卷积核进行2d卷积操作，再归一化
            )
        )
        del output
        # 步长为二，下采样，输出变小
        if self.stride == 2 and self.downAvg:
            return expanded

        # 如果输入和输出向量的维度相同，则加和再输出
        if expanded.size() == input.size():
            expanded = expanded + input
        return self.module_act(expanded)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=True),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=True),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # EESP(hidden_dim, hidden_dim),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=2, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)  # 16
        layers = [nn.Sequential(
            nn.Conv2d(1, output_channel, 3, 2, 1, bias=True),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel   # 16

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(

            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))

        )
        input_channel = output_channel

        output_channel = 512
        self.classifier = nn.Sequential(
            # nn.GRUCell(input_channel, output_channel),
            nn.Linear(input_channel, output_channel, bias=True),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghost_net(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, s
        [3, 64, 32, 0, 2],
        [3, 128, 64, 0, 1],
        # [5, 72, 40, 0, 2, 0],
        # [5, 120, 40, 0, 1, 0],
        [5, 256, 128, 1, 2],
        [5, 256, 128, 1, 1],
        # [3, 512, 256, 1, 1],
        # [5, 512, 256, 1, 1],
        # [5, 480, 112, 1, 1],
        # [5, 672, 112, 1, 1],
        # [5, 672, 160, 1, 2],
        # [5, 960, 160, 1, 1],
        # [7, 960, 160, 1, 1],
        # [5, 960, 160, 0, 1],
        # [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__=='__main__':
    from torchsummary import summary
    # from thop import profile

    model = ghost_net()
    model.eval()
    summary(model.cuda(),(1, 99, 39))
    # print(model)
    # input = torch.randn(32,3,224,224)
    # flops, params = profile(model, inputs=(input,))
    # y = model(input)
    # print(params)