import math

import torch
from torch import nn


class GhostModule1d(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule1d, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(nn.Conv1d(inp, init_channels, kernel_size, stride,
                                                    kernel_size // 2, bias=False),
                                          nn.BatchNorm1d(init_channels),
                                          nn.ReLU(inplace=True) if relu else nn.Sequential())

        self.cheap_operation = nn.Sequential(nn.Conv1d(init_channels, new_channels, dw_size, 1,
                                                       dw_size // 2, groups=init_channels, bias=False),
                                             nn.BatchNorm1d(new_channels),
                                             nn.ReLU(inplace=True) if relu else nn.Sequential())

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :]


class GhostModule1d_T(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule1d_T, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(nn.ConvTranspose1d(inp, init_channels, kernel_size, stride,
                                                             kernel_size // 2, bias=False, output_padding=1),
                                          nn.BatchNorm1d(init_channels),
                                          nn.ReLU(inplace=True) if relu else nn.Sequential())

        self.cheap_operation = nn.Sequential(nn.ConvTranspose1d(init_channels, new_channels, dw_size, 1,
                                                                dw_size // 2, groups=init_channels, bias=False),
                                             nn.BatchNorm1d(new_channels),
                                             nn.ReLU(inplace=True) if relu else nn.Sequential())

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :]


class AutoEncoder(nn.Module):
    def __init__(self, width=6000, height=1, channel=1):
        super(AutoEncoder, self).__init__()
        self.width = width
        self.height = height
        self.channel = channel

        inp = self.channel * self.width * self.height

        self.encoder = nn.Sequential(
            # GhostModule1d(self.channel, 64, 1, 2, 3, 2),
            # GhostModule1d(64, 32, 1, 2, 3, 2),
            # GhostModule1d(32, 16, 1, 2, 3, 2),
            # GhostModule1d(32, 16, 1, 2, 3),
            nn.Conv1d(self.channel, 64, 3, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 32, 3, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(32, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            # GhostModule1d_T(16, 32, 5, 2, 3),
            # GhostModule1d_T(16, 32, 1, 2, 3, 2),
            # GhostModule1d_T(32, 64, 1, 2, 3, 2),
            # GhostModule1d_T(64, self.channel, 1, 2, 3, 2),
            # nn.Linear(750, 6000),
            nn.ConvTranspose1d(16, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, self.channel, 3, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == '__main__':
    from torchsummary import summary

    # from thop import profile

    model = AutoEncoder()
    model.eval()
    summary(model.cuda(), (1, 6000))
