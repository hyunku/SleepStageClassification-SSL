# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class MultiMAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MAE_Encoder(nn.Module):
    def __init__(self,
                 input_size: int = 3000,
                 patch_size: int = 30,
                 emb_dim: int = 192,
                 num_layer: int = 8,
                 num_head: int = 3,
                 mask_ratio: float = 0.75):
        super().__init__()
        self.input_size = input_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

    def forward(self, x: torch.Tensor):
        return x


if __name__ == '__main__':
    pass
