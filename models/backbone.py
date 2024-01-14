# -*- coding:utf-8 -*-
import sys

import torch
from math import ceil
from models.utils import *
import warnings


class BaseCNN(nn.Module):
    def __init__(self, sampling_rate, stride, dropout):
        super(BaseCNN, self).__init__()
        self.input_time_length = sampling_rate * 30
        input_channels = 1
        kernel_size = int(sampling_rate // 4)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.final_length = self.get_final_length()

    def forward(self, x):
        b = x.shape[0] # 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) # 1, 128, 127
        x = torch.reshape(x, [b, -1]) # 1, 16256
        return x

    def get_final_length(self):
        x = torch.randn((1, 1, self.input_time_length))
        x = self.forward(x)
        x = torch.reshape(x, [1, -1])
        return x.shape[-1]


class ZZLet(nn.Module):
    def __init__(self, sampling_rate, stride, cnn_dropout, rnn_dropout, rnn_dim, rnn_layers=2):
        super().__init__()
        self.cnn_dropout = cnn_dropout
        self.rnn_dropout = rnn_dropout
        self.sampling_rate = sampling_rate
        self.rnn_dim = rnn_dim

        kernel_size = int(sampling_rate // 8)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(cnn_dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.cnn_o_size = self.cnn_out_size()
        # print(self.cnn_o_size)
        self.rnn = nn.LSTM(input_size=self.cnn_o_size, hidden_size=self.rnn_dim, num_layers=rnn_layers,
                           dropout=self.rnn_dropout, bidirectional=True)
        self.final_length = self.get_final_length()
        print(self.final_length)

    def forward(self, x):
        b = x.size()[0]
        # Convolution Neural Network
        cnn_outs = []
        for sample_x in torch.split(x, split_size_or_sections=self.sampling_rate, dim=-1):
            # print(sample_x.shape)
            x = self.conv1(sample_x)
            x = self.conv2(x)
            x = self.conv3(x)
            cnn_out = x.view([b, -1])
            cnn_outs.append(cnn_out)
        cnn_outs = torch.stack(cnn_outs, dim=1)

        # Recurrent Neural Network
        rnn_outs, _ = self.rnn(cnn_outs)

        # Skip-Connected
        out = rnn_outs[:, :, :self.rnn_dim] + rnn_outs[:, :, self.rnn_dim:]
        out = out.view([b, -1])
        return out

    def cnn_out_size(self):
        x = torch.randn((1, 1, self.sampling_rate))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, [-1])
        return x.shape[-1]

    def get_final_length(self):
        x = torch.randn((1, 1, self.sampling_rate * 30))
        x = self.forward(x)
        return x.shape[-1]


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class CNNEncoder2D_SLEEP(nn.Module):
    def __init__(self, n_dim, sampling_rate):
        super(CNNEncoder2D_SLEEP, self).__init__()
        self.sampling_rate = sampling_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(in_channels=6, out_channels=8, stride=2, downsample=True, pooling=False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

        self.sup = nn.Sequential(
            nn.Linear(128, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 5, bias=True),
        )

        self.final_length = self.get_final_length()


    def torch_stft(self, x):
        signal = [] # store STFT result
        warnings.filterwarnings("ignore")

        for s in range(x.shape[1]): # raw data is 3D -> (batch, num_signal(chan), len_signal)
            spectral = torch.stft(x[:, s, :], # roop each signal
                                  n_fft=256, # FFT size
                                  hop_length=256 * 1 // 4, # FFT size bin(FFT size 겹칠 크기) # default : 4
                                  center=False, # edge에서 0으로 padding 하지 않음.
                                  onesided=True, # 양수부분에서 정의된 STFT 결과값만 가져옴(절반만 가져옴)
                                  return_complex=False) # Ture : return input dim + 2 -> 실수부 + 허수부, False: return input dim + 1 -> 실수부
            signal.append(spectral)

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def get_final_length(self):
        x = torch.randn((1, 1, self.sampling_rate * 30))
        x = self.forward(x)
        return x.shape[-1]

    def forward(self, x, sup=False):
        x = self.torch_stft(x) # (1024, 1, 3000) -> (1024, 2, 129, 43) # batch, real/imaginary, Hz, time
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # 1, 32, 4, 2 -> b, c, h, w
        x = x.reshape(x.shape[0], -1) # 1, 256

        if sup: # supervised 인 경우 -> fine tunuing
            return self.sup(x)
        else:
            return x


class EncoderSequential(nn.Sequential):
    def __getattr__(self, name):
        if name == "encoder": # get_final_length를 self.encoder.encoder 에서도 사용하기 위한 메소드
            return self
        return super().__getattr__(name)


class BENDREncoder(nn.Module):
    def __init__(self, sampling_rate=100, in_features=1, encoder_h=256, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0., enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__()
        self.encoder_h = encoder_h
        self.sampling_rate = sampling_rate

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("conv{}".format(i+1), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout2d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

        self.final_length = self.get_final_length()

    def get_final_length(self):
        x = torch.randn((1, 1, self.sampling_rate * 30))
        x = self.forward(x)
        x = x.reshape(x.shape[0], -1)
        return x.shape[-1]


    def forward(self, x):
        return self.encoder(x)


# class BENDR_Encoder(nn.Module):
#     def __init__(self, sampling_rate, stride, cnn_dropout, rnn_dropout, rnn_dim, rnn_layers=2):
#         super().__init__()
#         self.sampling_rate = sampling_rate
#         self.conv1 = EncodingBlock()




if __name__ == '__main__':
    data = torch.randn(size=(300, 1, 3000))
    model = CNNEncoder2D_SLEEP(n_dim=100, sampling_rate=100)
    # for name, module in model.named_modules():
    #     if name in ['Encoder_0', 'Encoder_1']:
    #         module.eval()
    # print(model)
    # model = ZZLet(sampling_rate=100, stride=3, rnn_dim=100, cnn_dropout=0.35, rnn_dropout=0.35, rnn_layers=2)
    y = model(data)
    print(y.shape)
