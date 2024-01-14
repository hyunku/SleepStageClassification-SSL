# -*- coding:utf-8 -*-
import sys

import torch
import numpy as np
import torch.nn as nn
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from einops.layers.torch import Rearrange
import random
import torch.nn.functional as F

# mask_token : 어떤 값으로 마스킹을 할지(masking value), ex) 0
# mask : 마스킹 시킨 부분의 인덱스들이 담긴 리스트
# 순서: 데이터 -> class token add -> position embedding -> masking 순서임.
class Encoder(nn.Module):
    def __init__(self,
                 input_size: int = 3000,
                 patch_size: int = 10,
                 emb_dim: int = 192,
                 num_layer: int = 8,
                 num_head: int = 3,
                 masking_ratio: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        assert input_size % patch_size == 0, 'input_size must be divisible by patch_size'
        self.patch_num = self.input_size // self.patch_size
        self.masking_ratio = masking_ratio
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num + 1, emb_dim)) # position emb size는 패치에 토큰 더한것과 동일
        self.cls_token = nn.Parameter(torch.randn(1, emb_dim))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) # 1,1,192 가 0으로 채워져있는 텐서 -> 채울 값(0) 의미
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.patch = torch.nn.Conv2d(1, emb_dim, (1, patch_size), (1, patch_size))
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)


    @staticmethod
    def random_mask_patches(data, masking_ratio, mask_token):
        batch_size, patch_num, feature_dim = data.size()
        num_patches = int(patch_num * masking_ratio)

        masked_data = data.clone()
        mask_indices = []  # 마스킹된 인덱스 추적을 위한 리스트
        mask = torch.zeros_like(data)  # Create a mask with the same shape as data

        for i in range(batch_size):
            indices = random.sample(range(patch_num), num_patches)
            mask_indices.append(indices)  # 마스킹된 인덱스 추가
            for j in indices:
                # 패치 영역을 마스킹
                masked_data[i, j] = mask_token
                mask[i, j] = 1  # Update the mask

        return masked_data, mask_indices, mask

    @staticmethod
    def random_mask_patches_except_cls(data, masking_ratio, mask_token):
        batch_size, patch_num, feature_dim = data.size()
        num_patches = int((patch_num - 1) * masking_ratio)  # 클래스 토큰 제외

        masked_data = data.clone()
        mask_indices = []  # 마스킹된 인덱스 추적을 위한 리스트
        mask = torch.zeros_like(data)  # 마스크 생성

        for i in range(batch_size):
            # 클래스 토큰을 제외한 패치 중에서 무작위로 선택
            indices = random.sample(range(1, patch_num), num_patches) # 0은 클래스토큰으로 제외
            mask_indices.append(indices)  # 마스킹된 인덱스 추가
            for j in indices:
                # 패치 영역을 마스킹
                masked_data[i, j] = mask_token
                mask[i, j] = 1  # Update the mask

        return masked_data, mask_indices, mask


    @staticmethod
    def restore_masked_patches(masked_data, mask_indices, restore_value):
        restored_data = masked_data.clone()

        for i, indices in enumerate(mask_indices):
            for j in indices:
                # 마스킹된 패치를 원래 값으로 복원
                restored_data[i, j] = restore_value[i, j]

        return restored_data

    def forward(self, x: torch.Tensor, masking: bool):
        if masking:
            patches = self.patch(x) # x(raw): (256, 1, 1, 3000) -> (256, 192, 1, 300)
            patches = rearrange(patches, 'b c h w -> b (h w) c')  # patches : (256, 300, 192)
            batch_size, _, _ = patches.shape  # 256, 300, 192
            patches_plus_token = torch.cat([self.cls_token.repeat(batch_size, 1, 1), patches], dim=1)  # cls_token : (1, 192) -> batch만큼 repeat -> (256,1,192) -> (B, N+1, D)
            patches_vit_input = patches_plus_token + self.pos_embedding
            masked, mask_indices, mask = self.random_mask_patches_except_cls(patches_vit_input, self.masking_ratio, self.mask_token)

            patches_vit_output = self.layer_norm(self.transformer(masked))  # vit input과 output shape 동일한 것까지 확인함
            return patches_vit_output, mask_indices, mask
        else:
            patches = self.patch(x) # x(raw): (256, 1, 1, 3000)
            patches = rearrange(patches, 'b c h w -> b (h w) c')  # patches : (256, 300, 192)
            return patches


class SimMIM(nn.Module):
    def __init__(self,
                 input_size: int = 3000,
                 patch_size: int = 10,
                 emb_dim: int = 192,
                 num_layer: int = 8,
                 num_head: int = 3,
                 masking_ratio: float = 0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder = Encoder(
            input_size=input_size,
            patch_size=patch_size,
            emb_dim=self.emb_dim,
            num_layer=num_layer,
            num_head=num_head,
            masking_ratio=masking_ratio)

        self.decoder = nn.Conv1d(
                in_channels=self.emb_dim,
                out_channels=self.emb_dim,
                kernel_size=1,
                stride=1,
                padding=0)

    def forward(self, x):
        z, masked_indices, mask = self.encoder(x, masking=True)
        x_target = self.encoder(x, masking=False) # 256, 300, 192
        z = rearrange(z, 'b n d -> n b d')
        z = z[1:]
        mask = rearrange(mask, 'b n d -> n b d')
        mask = mask[1:]
        cls_token = z[:1]
        z = rearrange(z, 'n b d -> b n d')
        mask = rearrange(mask, 'n b d -> b n d')
        cls_token = rearrange(cls_token, 'n b d -> b n d')
        z = rearrange(z, 'b n d -> b d n')
        x_pred = self.decoder(z)
        x_pred = rearrange(x_pred, 'b d n -> b n d')

        return x_target, x_pred, mask


if __name__ == '__main__':
    m = SimMIM()
    d = torch.randn(size=(256, 1, 1, 3000))  # B, C, sample(data)
    target, pred, mask = m(d) # (256, 300, 192) -> (b,n,d) -> (batch, patch_num, feature_dim)
    loss_reconstruct = F.l1_loss(target, pred, reduction='none')
    loss = (loss_reconstruct * mask).sum() / (mask.sum() + 1e-5) # loss_recon * mask -> masking된 부분의 손실만 고려 -> mask의 sum 으로 나눠줌으로써 masking된 부분의 손실의 평균값을 구할수 있음
    print(loss)
