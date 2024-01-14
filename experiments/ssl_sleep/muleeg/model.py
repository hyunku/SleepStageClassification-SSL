# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from experiments.ssl_sleep.muleeg.encoder import Encoder
from typing import Tuple


class SleepModel(nn.Module):
    def __init__(self, sampling_rate, hidden_dim: int = 256):
        super(SleepModel, self).__init__()
        self.sampling_rate = sampling_rate
        self.backbone = Encoder(sampling_rate=sampling_rate)
        self.length = self.get_encoder_length()
        self.weak_pj1 = ProjectionHead(input_dim=self.length, hidden_dim=hidden_dim)
        self.weak_pj2 = ProjectionHead(input_dim=self.length * 2, hidden_dim=hidden_dim * 2)
        self.weak_pj3 = ProjectionHead(input_dim=self.length, hidden_dim=hidden_dim)

        self.strong_pj1 = ProjectionHead(input_dim=self.length, hidden_dim=hidden_dim)
        self.strong_pj2 = ProjectionHead(input_dim=self.length * 2, hidden_dim=hidden_dim * 2)
        self.strong_pj3 = ProjectionHead(input_dim=self.length, hidden_dim=hidden_dim)

    def get_encoder_length(self):
        input_times = torch.randn((1, 1, self.sampling_rate * 30))
        f = self.backbone(input_times)
        return int(f.shape[-1] / 2)

    def forward(self, weak_data: torch.Tensor, strong_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                   torch.Tensor, torch.Tensor,
                                                                                   torch.Tensor, torch.Tensor]:
        weak_eeg_data, strong_eeg_data = weak_data.float(), strong_data.float()

        weak_feats, strong_feats = self.backbone(weak_eeg_data), self.backbone(strong_eeg_data)
        size = int(weak_feats.shape[-1] / 2)
        w_t_feats, w_s_feats = weak_feats[:, :size], weak_feats[:, size:]
        s_t_feats, s_s_feats = strong_feats[:, :size], strong_feats[:, size:]

        w_f_feats = torch.cat((w_t_feats, w_s_feats), dim=-1)
        w_f_feats = self.weak_pj2(w_f_feats.unsqueeze(1))
        w_t_feats = self.weak_pj1(w_t_feats.unsqueeze(1))
        w_s_feats = self.weak_pj3(w_s_feats.unsqueeze(1))

        s_f_feats = torch.cat((s_t_feats, s_s_feats), dim=-1)
        s_f_feats = self.strong_pj2(s_f_feats.unsqueeze(1))
        s_t_feats = self.strong_pj1(s_t_feats.unsqueeze(1))
        s_s_feats = self.strong_pj3(s_s_feats.unsqueeze(1))

        return w_t_feats, w_f_feats, w_s_feats, \
               s_t_feats, s_f_feats, s_s_feats


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.projection_head(x)
        return x


if __name__ == '__main__':
    sm = SleepModel(sampling_rate=100)
    # torch.
