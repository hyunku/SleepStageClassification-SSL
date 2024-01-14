# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from typing import List, Tuple


class Encoder(nn.Module):
    def __init__(self, sampling_rate):
        super(Encoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.time_model = BaseNet()
        self.attention = Attention()
        self.spectral_model = CNNEncoder2DSLEEP(256)
        self.final_length = self.get_final_length()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = self.time_model(x)
        time_feats = self.attention(time)
        spectral_feats = self.spectral_model(x)
        feature = torch.cat([time_feats, spectral_feats], dim=-1)
        return feature

    def get_final_length(self):
        x = torch.randn(size=(1, 1, self.sampling_rate * 30))
        x = self.forward(x)
        return x.shape[-1]


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.att_dim = 256
        self.W = nn.Parameter(torch.randn(256, self.att_dim))
        self.V = nn.Parameter(torch.randn(self.att_dim, 1))
        self.scale = self.att_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        e = torch.matmul(x, self.W)
        e = torch.matmul(torch.tanh(e), self.V)
        e = e * self.scale
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        return x


# 1. Raw EEG
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False
    )


class BasicBlock_Bottle(nn.Module):
    expansion = 4

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock_Bottle, self).__init__()
        self.conv1 = nn.Conv1d(inplanes3, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=25, stride=stride, padding=12, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv3 = nn.Conv1d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )  #
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BaseNet(nn.Module):
    def __init__(self, input_channels=1, layers: List = [3, 4, 6, 3]):
        self.inplanes3 = 16
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=71, stride=2, padding=35, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=71, stride=2, padding=35)

        self.layer3x3_1 = self._make_layer3(BasicBlock_Bottle, 8, layers[0], stride=1)
        self.layer3x3_2 = self._make_layer3(BasicBlock_Bottle, 16, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock_Bottle, 32, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock_Bottle, 64, layers[3], stride=2)

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes3,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        x = self.layer3x3_4(x)
        return x


# 2. STFT Spectrogram
class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: bool = False,
            pooling: bool = False,
    ) -> None:
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the block."""

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual

        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)

        return out


class CNNEncoder2DSLEEP(nn.Module):
    def __init__(self, n_dim: int) -> None:
        super(CNNEncoder2DSLEEP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    @staticmethod
    def torch_stft(x_train: torch.Tensor) -> torch.Tensor:
        signal = list()

        for s in range(x_train.shape[1]):
            spectral = torch.stft(
                x_train[:, s, :],
                n_fft=256,
                hop_length=256 * 1 // 4,
                center=False,
                onesided=True,
                return_complex=False,
            )
            signal.append(spectral)

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat(
            [
                torch.log(torch.abs(signal1) + 1e-8),
                torch.log(torch.abs(signal2) + 1e-8),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x


if __name__ == '__main__':
    e = Encoder()
    e.forward(torch.randn(size=(1, 1, 3000)))