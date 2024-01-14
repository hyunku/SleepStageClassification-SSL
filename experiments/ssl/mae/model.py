# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from einops.layers.torch import Rearrange


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences: torch.Tensor, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        t, b, c = patches.shape
        remain_t = int(t * (1 - self.ratio))

        indexes = [random_indexes(t) for _ in range(b)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1),
                                          dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1),
                                           dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_t]
        return patches, forward_indexes, backward_indexes


class MAE_Encoder(nn.Module):
    def __init__(self,
                 input_size: int = 500,
                 patch_size: int = 10,
                 emb_dim: int = 192,
                 num_layer: int = 8,
                 num_head: int = 3,
                 mask_ratio: float = 0.75):
        super().__init__()
        self.input_size = input_size
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((input_size // patch_size), 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.patch = torch.nn.Conv2d(1, emb_dim, (1, patch_size), (1, patch_size))
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, x: torch.Tensor):
        patches = self.patch(x)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches, _, restore_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        return features, restore_indexes


class MAE_Decoder(nn.Module):
    def __init__(self,
                 input_size: int = 500,
                 patch_size: int = 10,
                 emb_dim: int = 192,
                 num_layer: int = 4,
                 num_head: int = 3):
        super().__init__()
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.pos_embedding = torch.nn.Parameter(torch.zeros((input_size // patch_size) + 1, 1, emb_dim))
        self.head = torch.nn.Linear(emb_dim, patch_size)
        self.patch2signal = Rearrange('h b p -> b (h p)', p=patch_size, h=input_size // patch_size)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features: torch.Tensor, restore_indexes: torch.Tensor):
        t = features.shape[0]
        restore_indexes = torch.cat([torch.zeros(1, restore_indexes.shape[1]).to(restore_indexes),
                                     restore_indexes+1], dim=0)
        features = torch.cat([features, self.mask_token.expand(restore_indexes.shape[0] - features.shape[0],
                                                               features.shape[1], -1)], dim=0)
        features = take_indexes(features, restore_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[t:] = 1
        mask = take_indexes(mask, restore_indexes[1:] - 1)
        signal, mask = self.patch2signal(patches), self.patch2signal(mask)
        return signal, mask


class MAE_ViT(nn.Module):
    def __init__(self,
                 input_size: int = 3000,
                 patch_size: int = 10,
                 emb_dim: int = 192,
                 encoder_layer: int = 12,
                 encoder_head: int = 3,
                 decoder_layer: int = 4,
                 decoder_head: int = 3,
                 mask_ratio: int = 0.75):
        super().__init__()
        self.encoder = MAE_Encoder(input_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(input_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, x):
        features, restore_indexes = self.encoder(x)
        predicted_signal, mask = self.decoder(features, restore_indexes)

        return predicted_signal, mask


if __name__ == '__main__':
    pass
