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

# TODO: 이미지를 넣어보자!!!!

# TODO: transformer에 넣을 데이터 준비과정
# TODO: 1. 패치화  2. 클래스 토큰 추가  3.포지션 임베딩  4.마스킹(단, 클래스 토큰은 마스킹X)

# TODO: Multimodal의 경우
# TODO: 1. 패치화(=patch embedding) 2. class token, seperable token 추가 3. 각각 다른 position embedding 진행 4. 여러 representation vector들 concat 5. masking(단, class, seperate 토큰은 masking X)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Encoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=256, num_layer=8, num_heads=4, mlp_ratio=4.):
        super(Encoder, self).__init__()
        assert img_size % patch_size == 0, 'img_size must be divisible by patch_size'
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim # vit 내부 dim
        self.num_layer = num_layer # vit block 몇번?
        self.num_heads = num_heads # msa head 몇개?
        self.mlp_ratio = mlp_ratio # msa head들 합한 후 mlp의 hidden dim(embed dim * mlp_ratio) 의 비율
        self.norm = nn.LayerNorm(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=True)

        # self.patch = torch.nn.Conv2d(1, self.embed_dim, (1, self.patch_size), (1, self.patch_size))
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.Sequential(*[Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(self.num_layer)])


        self.init_weight()

    def init_weight(self):
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d) => Patch embedding 초기화
        w = self.patch.weight.data
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


    def forward(self, x, mask_ratio=0.9):
        mask_ratio = float(mask_ratio)
        # embed patches # raw (b, 3, 32, 32)
        x = self.patch(x) # (64, 3, 32, 32) -> (64, 256, 8, 8) = ( B, D, P_h, P_w )
        x = rearrange(x, 'b c h w -> b (h w) c') # (64, 64, 256) = ( B, N, D )

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] # (64, 64, 256) = ( B, N, D )

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio) # x : 마스킹시키지 않을 패치들 -> ( 64, 6, 256 )

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # ( 1, 1, D) -> (1, 1, 256)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # ( B, 1, D ) -> ( 64, 1, 256)
        x = torch.cat((cls_tokens, x), dim=1) # ( B, N+1, D ) -> (64, 29+1, 256)

        # apply Transformer blocks
        x = self.norm(self.transformer(x)) # (64, 29+1, 256)

        if mask_ratio > 0.0 : # train (SSL) mode
            return x, mask, ids_restore

        elif mask_ratio == 0.0:
            cls_token = x[:,0,:] # (64, 256)
            return cls_token


class Decoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=256, dec_embed_dim=128, dec_num_layer=8, dec_num_heads=4, mlp_ratio=4.):
        super(Decoder, self).__init__()
        assert img_size % patch_size == 0, 'img_size must be divisible by patch_size'
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim # encoder output dim
        self.dec_embed_dim = dec_embed_dim # decoder dim
        self.dec_num_layer = dec_num_layer # decoder vit block 몇번?
        self.dec_num_heads = dec_num_heads # decoder msa head 몇개?
        self.mlp_ratio = mlp_ratio # msa head들 합한 후 mlp의 hidden dim(embed dim * mlp_ratio) 의 비율
        self.dec_norm = nn.LayerNorm(self.dec_embed_dim)

        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.dec_embed_dim), requires_grad=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))

        self.dec_embed = nn.Linear(self.embed_dim, self.dec_embed_dim, bias=True)
        self.transformer = nn.Sequential(*[Block(self.dec_embed_dim, self.dec_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(self.dec_num_layer)])
        self.dec_proj = nn.Linear(self.dec_embed_dim, self.patch_size**2 * 3, bias=True) # decoder to patch

        self.init_weight()

    def init_weight(self):
        pos_embed = get_2d_sincos_pos_embed(self.dec_embed_dim, int(self.num_patches ** .5), cls_token=True)
        self.dec_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) => cls_token 초기화
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, x, mask, ids_restore): # ids_restore : (b, 64) 원본 이미지 총 패치 수 64개
        x = self.dec_embed(x) # (b, 6+1, 256) -> (b, 6+1, 128)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # 원본 패치 개수(300) - 마스킹시키지않은패치개수(29+1(클래스토큰)) + 토큰개수(1) -> 마스킹시킬 패치들(64, 271, 128)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token -> (64, 29+271, 128) -> x_는 원본 이미지 패치 -> 남긴 패치들 - 1 + 마스크 영역들 합쳐서 원본 이미지와 동일하게 size만듬
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle -> 아까 전체 패치에 대한 index를 저장해놓았었음 -> 단순하게 합친 patch + 마스크들로 원본 이미지로 복원
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token -> 원본 이미지 패치 ( 64, 300, 128 ) -> (64, 300+1, 128)

        # add pos embed
        x = x + self.dec_pos_embed

        # apply Transformer blocks
        x = self.dec_norm(self.transformer(x)) # (64, 300+1, 128)

        # predictor projection
        x = self.dec_proj(x) # ( b, 64+1, 4^2*3)

        # remove cls token
        pred_patch = x[:, 1:, :] # (b, 64, 4^2*3)

        return pred_patch


class MAE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=256, num_layer=8, num_heads=4,
                 dec_embed_dim=128, dec_num_layer=8, dec_num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        self.encoder = Encoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, num_layer=num_layer, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.decoder = Decoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, dec_embed_dim=dec_embed_dim, dec_num_layer=dec_num_layer, dec_num_heads=dec_num_heads, mlp_ratio=mlp_ratio)

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

    def patchify(self, imgs): # (b c h w) -> (b n p^2*c)
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % self.patch_size == 0 # 정사각형이 맞는지 체크

        p = self.patch_size
        p_h = p_w = self.img_size // self.patch_size
        x = imgs.reshape(shape=(imgs.shape[0], 3, p_h, p, p_w, p))
        x = rearrange(x, 'b c p_h p1 p_w p2 -> b (p_h p_w) (p1 p2 c)')
        return x

    def unpatchify(self, x): # (b n p^2*c) -> (b 3 h w)
        b, n, d = x.shape
        p = self.patch_size
        p_h = p_w = int(n ** .5) # route
        assert p_h * p_w == n

        x = x.reshape(shape=(b, p_h, p_w, p, p, 3))
        x = rearrange(x, 'b p_h p_w p1 p2 c -> b c (p_h p1) (p_w p2)')
        return x


    def forward(self, x, mask_ratio=0.9):
        remain_x, mask, ids_restore = self.encoder(x, mask_ratio)
        pred_patch = self.decoder(remain_x, mask, ids_restore)
        target_patch = self.patchify(x)

        error = (pred_patch - target_patch) ** 2
        loss = error.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        pred_img = self.unpatchify(pred_patch)
        origin_img = x

        return loss, origin_img, pred_img


if __name__ == '__main__':
    # m1 = Encoder()
    # m2 = Decoder()
    m = MAE()
    d = torch.randn(size=(11,3,32,32)) # image
    loss, origin, pred = m(d)
    print(loss)
    # for name, module in m.named_modules():
    #     print(name)
    # remain_x, mask, ids_restore = m1(d) # remain_x : (64, 29+1, 256) mask : (64, 300) -> 1은 남길 패치, 0은 마스킹시킬 패치
    # out = m2(remain_x, ids_restore, mask)
    # print(out.shape)
    # print(mask.shape)
    # pred_sig, mask_sig = m(d, mask_ratio=0.9)
    # print(pred_sig.shape)
    # print(pred_sig.squeeze().shape)
    # print(mask_sig.shape)
    # print(mask_sig.squeeze().shape)


#
#     # target, pred, mask = m(d) # (256, 300, 192) -> (b,n,d) -> (batch, patch_num, feature_dim)
#     loss_reconstruct = F.l1_loss(target, pred, reduction='none')
#     loss = (loss_reconstruct * mask).sum() / (mask.sum() + 1e-5) # loss_recon * mask -> masking된 부분의 손실만 고려 -> mask의 sum 으로 나눠줌으로써 masking된 부분의 손실의 평균값을 구할수 있음
#     print(loss)
#     # d = torch.randn(size=(256,1,1,1500))
#     # m = PatchData()
#     # output = m(d)
#     # print(output)