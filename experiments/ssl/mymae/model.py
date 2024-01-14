import sys
from typing import Tuple

import torch
from math import ceil
from models.utils import *
import warnings
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
import random
import torch.nn.functional as F

# TODO: emb/enc 차이 => position_encoding: 단순 벡터값 더해주는거 -> requires_grad=False or True, position_embedding: input을 latent space에 일일이 다 넣어서 적합한 공간에 넣음 -> requires_grad=True
# TODO: 데이터 전처리 정규화 (global normalization, Z-normalize) -> 데이터셋의 분포 정규화 F.normalize -> 벡터의 크기 정규화 ( 방향 그대로, 크기만 조절 )

# TODO: 할것들 정리: 1. 변수명 재정의

def sincos_pos_embed_1D(embed_dim, pos, cls_token=False):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos, dtype=np.float32).reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class Encoder(nn.Module):
    def __init__(self, input_size=3000, patch_size=10, embed_dim=256, num_layer=8, num_heads=4, mlp_ratio=4.):
        super(Encoder, self).__init__()
        assert input_size % patch_size == 0, 'input_size must be divisible by patch_size'
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim # vit 내부 dim
        self.num_layer = num_layer # vit block 몇번?
        self.num_heads = num_heads # msa head 몇개?
        self.mlp_ratio = mlp_ratio # msa head들 합한 후 mlp의 hidden dim(embed dim * mlp_ratio) 의 비율
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = self.input_size // self.patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False)

        self.patch_emb = torch.nn.Conv2d(1, self.embed_dim, (1, self.patch_size), (1, self.patch_size))
        self.transformer = nn.Sequential(*[Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, qk_norm=True,
                                                 act_layer=nn.GELU, norm_layer=nn.LayerNorm) for _ in range(self.num_layer)])

        self.final_length = embed_dim # used for evaluation mode

        self.init_weight()

    def init_weight(self):
        # initialization

        # initialize (and freeze) pos_embed by sin-cos embedding => Position embedding 초기화
        pos_embed = sincos_pos_embed_1D(self.embed_dim, int(self.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d) => Patch embedding 초기화
        w = self.patch_emb.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) => cls_token 초기화
        trunc_normal_(self.cls_token, std=.02)


    def random_masking(self, x, mask_ratio): # 1은 masking 진행!! 0은 masking 진행 X!!
        B, N, D = x.shape  # batch, length, dim -> b, n, d
        len_keep = int(N * (1 - mask_ratio)) # 마스킹시키지 않을 마스크 개수

        noise = torch.rand(B, N, device=x.device)  # ( b, n ) -> 0~1사이값 -> ( 64, 187 )

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # 현재 dim=1(N) 기준 오름차순 정렬된 인덱스가 저장되어있음. 이 중 masking 시키지 않을 영역 선택
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # 전체 데이터에서 masking 시키지 않을 영역 선택해서 모음

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device) # ( b, n) -> 1로 이루어져있음
        mask[:, :len_keep] = 0 # 0은 마스킹X , 1은 masking

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # 1,0 로 이루어진 mask에서 복원한 인덱스들을 모음

        return x_masked, mask, ids_restore # x_masked : masking된 데이터(masked되지 않은 영역만 추출), mask : 0,1로된 마스크, ids_restore : 원래 순서로 되돌리는데 사용되는 인덱스


    def forward(self, x, mask_ratio=0.0):
        mask_ratio = float(mask_ratio)
        # raw (b, 1, 1, sample) -> (64, 1, 1, 3000)
        # L2 Normalization
        # x = F.normalize(x, p=2, dim=-1) # p:norm method ( 2 is L2 norm )
        x = self.patch_emb(x) # (64, 1, 1, 3000) -> (64, 256, 1, 300) = ( B, D, 1, N )
        x = rearrange(x, 'b c h w -> b (h w) c') # (64, 300, 256) = ( B, N, D )

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] # (64, 300, 256) = ( B, N, D )

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio) # x : 마스킹시키지 않을 패치들 -> ( 64, 29, 256 )

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # ( 1, 1, D) -> (1, 1, 256)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # ( B, 1, D ) -> ( 64, 1, 256)
        x = torch.cat((cls_tokens, x), dim=1) # ( B, N+1, D ) -> (64, 29+1, 256)

        # apply Transformer blocks
        x = self.norm(self.transformer(x)) # (64, 29+1, 256)

        x = F.normalize(x, dim=-1)

        if mask_ratio > 0.0 : # train (SSL) mode
            return x, mask, ids_restore

        elif mask_ratio == 0.0: # eval mode
            cls_token = x[:,0,:] # (64, 256)
            return cls_token


class Decoder(nn.Module):
    def __init__(self, input_size=3000, patch_size=10, embed_dim=256, dec_embed_dim=128, dec_num_layer=8, dec_num_heads=4, mlp_ratio=4.):
        super(Decoder, self).__init__()
        assert input_size % patch_size == 0, 'input_size must be divisible by patch_size'
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = self.input_size // self.patch_size
        self.embed_dim = embed_dim # encoder output dim
        self.dec_embed_dim = dec_embed_dim # decoder dim
        self.dec_num_layer = dec_num_layer # decoder vit block 몇번?
        self.dec_num_heads = dec_num_heads # decoder msa head 몇개?
        self.mlp_ratio = mlp_ratio # msa head들 합한 후 mlp의 hidden dim(embed dim * mlp_ratio) 의 비율
        self.dec_norm = nn.LayerNorm(self.dec_embed_dim)

        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.dec_embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))

        self.dec_embed = nn.Linear(self.embed_dim, self.dec_embed_dim, bias=True)
        self.transformer = nn.Sequential(*[Block(self.dec_embed_dim, self.dec_num_heads, self.mlp_ratio,
                                                 qkv_bias=True, qk_norm=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm) for _ in range(self.dec_num_layer)])
        self.dec_proj = nn.Linear(self.dec_embed_dim, self.patch_size, bias=True) # decoder to patch

        self.init_weight()

    def init_weight(self):
        # initialization

        # initialize (and freeze) pos_embed by sin-cos embedding => Position embedding 초기화
        pos_embed = sincos_pos_embed_1D(self.dec_embed_dim, int(self.num_patches), cls_token=True)
        self.dec_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) => cls_token 초기화
        trunc_normal_(self.mask_token, std=.02)


    def forward(self, x, mask, ids_restore): # mask, ids_restore : (64, 300) 원본 신호 패치화시키면 300개였었음
        x = self.dec_embed(x) # (64, 29+1, 256) -> (64, 29+1, 128)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # 원본 패치 개수(300) - 마스킹시키지않은패치개수(29+1(클래스토큰)) + 토큰개수(1) -> 마스킹시킬 패치들(64, 271, 128)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token -> (64, 29+271, 128) -> x_는 원본 이미지 패치 -> 남긴 패치들 - 1 + 마스크 영역들 합쳐서 원본 이미지와 동일하게 size만듬
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle -> 아까 전체 패치에 대한 index를 저장해놓았었음 -> 단순하게 합친 patch + 마스크들로 원본 이미지로 복원
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token -> 원본 이미지 패치 ( 64, 300, 128 ) -> (64, 300+1, 128)

        # add pos embed
        x = x + self.dec_pos_embed

        # apply Transformer blocks
        x = self.dec_norm(self.transformer(x)) # (64, 300+1, 128)

        # predictor projection
        x = self.dec_proj(x) # ( 64, 300+1, 10)

        # remove cls token
        pred_patch = x[:, 1:, :] # (64, 300, 10)
        mask_patch = mask.unsqueeze(-1).expand_as(pred_patch) # (64, 300, 1) -> (64, 300, 10) # 패치 기준 반복

        return pred_patch, mask_patch # (b, n, d)

class MAE(nn.Module):
    def __init__(self, input_size=3000, patch_size=10, embed_dim=256, num_layer=8, num_heads=4,
                 dec_embed_dim=128, dec_num_layer=8, dec_num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size

        self.encoder = Encoder(input_size=input_size, patch_size=patch_size, embed_dim=embed_dim, num_layer=num_layer, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.decoder = Decoder(input_size=input_size, patch_size=patch_size, embed_dim=embed_dim, dec_embed_dim=dec_embed_dim, dec_num_layer=dec_num_layer, dec_num_heads=dec_num_heads, mlp_ratio=mlp_ratio)
        self.final_length = embed_dim

        # initialize nn.Linear and nn.LayerNorm => 모델 전체의 layernorm과 linear 초기화
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x): # signal (b 1 1 3000) -> patch (b n d)
        p = self.patch_size
        n = self.input_size // self.patch_size
        x = x.squeeze() # b 1 1 3000 -> b 3000
        x = x.reshape(shape=(x.shape[0], n, p))
        return x # b n d

    def unpatchify(self, x): # patch ( b n d ) -> signal ( b 3000 )
        x = rearrange(x, 'b n d -> b (n d)')
        return x

    def forward(self, x, mask_ratio=0.0):
        if mask_ratio > 0.0:
            remain_x, mask, ids_restore = self.encoder(x, mask_ratio)
            pred_patch, mask_patch = self.decoder(remain_x, mask, ids_restore) # b n d
            origin_sig = x.squeeze() # b 1 1 s -> b s
            pred_sig = self.unpatchify(pred_patch) # b n d -> b s
            mask = self.unpatchify(mask_patch) # b n d -> b s
            return origin_sig, pred_sig, mask # b s

        elif mask_ratio == 0.0:
            cls_token = self.encoder(x, mask_ratio)
            return cls_token


if __name__ == '__main__':
    # m1 = Encoder()
    # m2 = Decoder()
    m = MAE()
    d = torch.randn(size=(64,1,1,3000))
    # for name, module in m.named_modules():
    #     print(name)
    # remain_x, mask, ids_restore = m1(d) # remain_x : (64, 29+1, 256) mask : (64, 300) -> 1은 남길 패치, 0은 마스킹시킬 패치
    # out = m2(remain_x, ids_restore, mask)
    # print(out.shape)
    # print(mask.shape)
    loss, origin_sig, pred_sig, mask = m(d, mask_ratio=0.9)
    print(loss)
    print(origin_sig.shape)
    print(pred_sig.shape)
    print(mask.shape)
