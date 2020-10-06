import math

import torch
import torch.nn as nn


"""
    该模块只能增加特征图的通道数，不能够改变其尺寸
    主要包括两步操作：
                1. 原始卷积操作，生成一定量m个特征图
                2. 廉价线性变换得到一定量s个冗余特征图
"""


class GhostModule:
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(nn.Conv2d(inp, init_channels, kernel_size, stride,
                                                    kernel_size // 2, bias=False),
                                          nn.BatchNorm2d(init_channels),
                                          nn.ReLU(inplace=True) if relu else nn.Sequential())

        self.cheap_operation = nn.Sequential(nn.Conv2d(init_channels, new_channels, dw_size, 1,
                                                       dw_size // 2, groups=init_channels, bias=False),
                                             nn.BatchNorm2d(new_channels),
                                             nn.ReLU(inplace=True) if relu else nn.Sequential())

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

