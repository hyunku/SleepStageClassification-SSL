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

# TODO: transformer에 넣을 데이터 준비과정
# TODO: 1. 패치화  2. 클래스 토큰 추가  3.포지션 임베딩  4.마스킹(단, 클래스 토큰은 마스킹X)

# TODO: Multimodal의 경우
# TODO: 1. 패치화(=patch embedding) 2. class token, seperable token 추가 3. 각각 다른 position embedding 진행 4. 여러 representation vector들 concat 5. masking(단, class, seperate 토큰은 masking X)

# For Spectogram Encoder
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


class SpectogramEncoder(nn.Module):
    def __init__(self, seq_len=3000, feature_dim=1500, sampling_rate=256, projection_hidden=3000, projection_type='nonlinear', fft_window=8):
        super(SpectogramEncoder, self).__init__()
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.feature_dim = feature_dim
        self.embedding_dim = projection_hidden
        self.projection_type = projection_type
        self.fft_window = fft_window

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(in_channels=6, out_channels=8, stride=2, downsample=True, pooling=False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        self.fc_in_features = self._get_conv_output()
        self.linear_projection = nn.Sequential(
            nn.Linear(self.fc_in_features, self.feature_dim, bias=True),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU()
        )
        self.nonlinear_projection = nn.Sequential(
            nn.Linear(self.fc_in_features, self.embedding_dim, bias=True),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU()
        )
        self.init_weight()

    def _get_conv_output(self):
        x = torch.randn(1, 1, self.seq_len)
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return x.size(1)

    def init_weight(self):
        if self.projection_type == 'linear':
            # nn.init.kaiming_normal_(self.linear_projection[0].weight, nonlinearity='relu')
            nn.init.xavier_uniform_(self.linear_projection[0].weight)
        elif self.projection_type == 'nonlinear':
            # nn.init.kaiming_normal_(self.nonlinear_projection[0].weight, nonlinearity='relu')
            # nn.init.kaiming_normal_(self.nonlinear_projection[3].weight, nonlinearity='relu')
            nn.init.xavier_uniform_(self.nonlinear_projection[0].weight)
            nn.init.xavier_uniform_(self.nonlinear_projection[3].weight)

    def torch_stft(self, x):
        signal = [] # store STFT result
        warnings.filterwarnings("ignore")

        for s in range(x.shape[1]): # raw data is 3D -> (batch, num_signal(chan), len_signal)
            spectral = torch.stft(x[:, s, :], # roop each signal
                                  n_fft=256, # FFT size
                                  hop_length=self.fft_window, # FFT size bin(FFT size 겹칠 크기)
                                  center=False, # edge에서 0으로 padding 하지 않음.
                                  onesided=True, # 양수부분에서 정의된 STFT 결과값만 가져옴(절반만 가져옴)
                                  return_complex=False) # Ture : return input dim + 2 -> 실수부 + 허수부, False: return input dim + 1 -> 실수부
            signal.append(spectral)

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x):
        x = self.torch_stft(x) # (1024, 1, 3000) -> (1024, 2, 129, 43) # batch, real/imaginary, Hz, time
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # 1, 32, 4, 2 -> b, c, h, w
        x = x.reshape(x.shape[0], -1)  # 1, 1408
        if self.projection_type == 'linear':
            x = self.linear_projection(x) # 1, 1500
        elif self.projection_type == 'nonlinear':
            x = self.nonlinear_projection(x)
        x = x.reshape(x.shape[0], 1, 1, -1)
        return x


# For Signal Encoder
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels)
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


class SignalEncoder(nn.Module):
    def __init__(self, seq_len=3000, feature_dim=1500, projection_hidden=3000, projection_type='nonlinear'):
        super(SignalEncoder, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.embedding_dim = projection_hidden
        self.projection_type = projection_type
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock1D(in_channels=6, out_channels=8, stride=2, downsample=True, pooling=False)
        self.conv3 = ResBlock1D(8, 16, 2, True, True)
        self.conv4 = ResBlock1D(16, 32, 2, True, True)
        self.fc_in_features = self._get_conv_output()
        self.linear_projection = nn.Sequential(
            nn.Linear(self.fc_in_features, self.feature_dim, bias=True),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU()
        )
        self.nonlinear_projection = nn.Sequential(
            nn.Linear(self.fc_in_features, self.embedding_dim, bias=True),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU()
        )
        self.init_weight()

    def init_weight(self):
        if self.projection_type == 'linear':
            # nn.init.kaiming_normal_(self.linear_projection[0].weight, nonlinearity='relu')
            nn.init.xavier_uniform_(self.linear_projection[0].weight)
        elif self.projection_type == 'nonlinear':
            # nn.init.kaiming_normal_(self.nonlinear_projection[0].weight, nonlinearity='relu')
            # nn.init.kaiming_normal_(self.nonlinear_projection[3].weight, nonlinearity='relu')
            nn.init.xavier_uniform_(self.nonlinear_projection[0].weight)
            nn.init.xavier_uniform_(self.nonlinear_projection[3].weight)

    def _get_conv_output(self):
        x = torch.randn(1, 1, self.seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return x.size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # 1, 32, 94
        x = x.reshape(x.shape[0], -1)  # 1, 3008
        if self.projection_type == 'linear':
            x = self.linear_projection(x) # 1, 1500
        elif self.projection_type == 'nonlinear':
            x = self.nonlinear_projection(x)
        x = x.reshape(x.shape[0], 1, 1, -1) # 1, 1, 1, 1500
        return x


class PatchData(nn.Module):
    def __init__(self,
                 feature_dim: int = 1500,
                 patch_size: int = 15,
                 emb_dim: int = 192):
        super(PatchData, self).__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        assert feature_dim % patch_size == 0, 'input_size must be divisible by patch_size'
        self.patch_num = self.feature_dim // self.patch_size
        self.sig_pos_embedding = nn.Parameter(torch.randn(1, self.patch_num + 1, self.emb_dim)) # cls token add pos embedding
        self.spe_pos_embedding = nn.Parameter(torch.randn(1, self.patch_num + 1, self.emb_dim)) # sep token add pos embedding
        # self.sig_type_embedding = nn.Parameter(torch.randn(1, 1, self.emb_dim)) # for type embedding
        # self.spe_type_embedding = nn.Parameter(torch.randn(1, 1, self.emb_dim)) # for type embedding
        self.cls_token = nn.Parameter(torch.randn(1, emb_dim))
        self.sep_token = nn.Parameter(torch.randn(1, emb_dim))
        self.patch = torch.nn.Conv2d(1, emb_dim, (1, patch_size), (1, patch_size))
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.sep_token, std=.02)
        trunc_normal_(self.sig_pos_embedding, std=.02)
        trunc_normal_(self.spe_pos_embedding, std=.02)
        # trunc_normal_(self.sig_type_embedding, std=.02)
        # trunc_normal_(self.spe_type_embedding, std=.02)

    def forward(self, z1, z2): # get ready for transformer input -> add token + position embedding
        p1 = self.patch(z1)  # x(raw): (256, 1, 1, 1500) -> (256, 192, 1, 100) -> Making patch
        p2 = self.patch(z2)
        p1 = rearrange(p1, 'b c h w -> b (h w) c')  # patches : (256, 100, 192) -> (B, N, D) -> patch embedding
        p2 = rearrange(p2, 'b c h w -> b (h w) c')
        batch_size = p1.shape[0]
        p1 = torch.cat([self.cls_token.repeat(batch_size, 1, 1), p1], dim=1) # cls_token : (1, 192) -> batch만큼 repeat -> (256,1,192) -> (B, N+1, D)
        p2 = torch.cat([self.sep_token.repeat(batch_size, 1, 1), p2], dim=1)

        p1_cls_pos_add = p1 + self.sig_pos_embedding
        p2_sep_pos_add = p2 + self.spe_pos_embedding
        # p1_cls_pos_add = p1_cls_pos_add + self.sig_type_embedding # type embedding
        # p2_sep_pos_add = p2_sep_pos_add + self.spe_type_embedding # type embedding
        patches_pos_emb = torch.cat([p1_cls_pos_add, p2_sep_pos_add], dim=1) # (B, 2N+2, D)
        sep_token_idx = p1.shape[1] # 클래스토큰을 달은 representation길이 바로 다음의 패치의 idx가 separable token의 인덱스가 될 것.
        return patches_pos_emb, sep_token_idx  # shape: B, N+1, D

    def only_patch(self, z1, z2):
        p1 = self.patch(z1)  # x(raw): (256, 1, 1, 1500) -> (256, 192, 1, 100) -> Making patch
        p2 = self.patch(z2)  # x(raw): (256, 1, 1, 1500) -> (256, 192, 1, 100) -> Making patch
        p1 = rearrange(p1, 'b c h w -> b (h w) c')  # patches : (256, 100, 192) -> (B, N, D) -> patch embedding
        p2 = rearrange(p2, 'b c h w -> b (h w) c')  # patches : (256, 100, 192) -> (B, N, D) -> patch embedding
        z_target = torch.cat([p1, p2], dim=1)
        return z_target



# Entire Model
class TSBendr(nn.Module):
    def __init__(self,
                 seq_len: int = 3000,
                 feature_dim: int = 1500,
                 projection_hidden: int = 3000,
                 projection_type: str = 'linear',
                 fft_window: int = 8,
                 patch_size: int = 10,
                 emb_dim: int = 192,
                 num_layer: int = 8,
                 num_head: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.sig_encoder = SignalEncoder(seq_len=seq_len, feature_dim=feature_dim, projection_hidden=projection_hidden, projection_type=projection_type)
        self.spe_encoder = SpectogramEncoder(seq_len=seq_len, feature_dim=feature_dim, projection_hidden=projection_hidden, projection_type=projection_type, fft_window=fft_window)
        self.patch = PatchData(feature_dim=feature_dim,
                               patch_size=patch_size,
                               emb_dim=emb_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) # 1,1,192 가 0으로 채워져있는 텐서 -> 채울 값(0) 의미
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.final_length = emb_dim

    def forward(self, x, masking_ratio=0.0):
        self.mask_token = self.mask_token.to(x.device)
        # check type
        masking_ratio = float(masking_ratio)

        # feature extract
        z_sig = self.sig_encoder(x)
        z_spe = self.spe_encoder(x) # 현재 둘다 256,1,1,1500

        # L2 Normalization -> 두개의 embedding 벡터들의 분포가 달라 끼치는 영향이 다를수 있음 -> 정규화
        z_sig = F.normalize(z_sig, p=2, dim=-1)
        z_spe = F.normalize(z_spe, p=2, dim=-1)

        # make patch + add token + position embedding
        patch, sep_token_idx = self.patch(z_sig, z_spe)

        if masking_ratio > 0.0: # 마스킹율이 0이 아니면 SSL(train) mode
            # masking
            masked_patches, _, mask = self.random_mask_patches_except_cls_sep(patch, masking_ratio, self.mask_token, sep_token_idx)

            # pass Transformer
            output = self.layer_norm(self.transformer(masked_patches))

            # make preds
            z_pred = torch.cat([output[:, 1:sep_token_idx, :], output[:, sep_token_idx+1:, :]], dim=1) # 256, 300, 192 -> delete class token and seperable token
            mask = torch.cat([mask[:, 1:sep_token_idx, :], mask[:, sep_token_idx+1:, :]], dim=1)

            # make target
            # z_target = self.patch.only_patch(z_sig, z_spe) # this is for non-pos embedding for target
            z_target = torch.cat([patch[:, 1:sep_token_idx, :], patch[:, sep_token_idx+1:, :]], dim=1) # this is for pos-embedding for target

            return z_target, z_pred, mask

        # 마스킹율이 0이면 evaluate mode -> 클래스토큰만 빼서 classification 진행
        elif masking_ratio == 0.0:
            out = self.layer_norm(self.transformer(patch)) # (B, N+2, D)
            cls_token = out[:, 0, :] # 클래스 토큰
            # cls_token = torch.mean(out[:, 1:, :], dim=1) # 클래스 토큰 제외 전체 feature들의 평균
            sep_token = out[:, sep_token_idx, :]
            cls_token = (cls_token + sep_token) / 2 # 클래스토큰과 sep 토큰의 평균 -> average feature
            return cls_token  # shape: 256, 192


    @staticmethod
    def random_mask_patches_except_cls_sep(data, masking_ratio, mask_token, separate_token_position):
        batch_size, patch_num, feature_dim = data.size()
        num_patches_to_mask = int((patch_num - 2) * masking_ratio)  # 클래스 토큰과 separate 토큰 제외

        # class token과 separate token을 제외한 인덱스 생성
        valid_indices = torch.ones(batch_size, patch_num, device=data.device).bool()  # True는 Masking 대상, 일단 True로 이루어진 텐서 생성
        valid_indices[:, [0, separate_token_position]] = 0  # 0번인덱스인 클래스 토큰과 sep 토큰은 0(False)으로 만들어줌

        # 각 배치마다 무작위로 인덱스를 선택
        rand_indices = torch.multinomial(valid_indices.float(), num_patches_to_mask, replacement=False)  # validices는 현재 Bool 타입 -> 1,0 으로 이루어진 텐서로 변환

        # 마스크 생성
        mask = torch.zeros(batch_size, patch_num, dtype=torch.bool, device=data.device)  # False로 이루어진 텐서 생성
        mask.scatter_(1, rand_indices, True)  # 각 배치별로(1) 선택된 인덱스(rand_indices)에 True를 부여

        # 마스킹된 위치에 mask_token을 적용하고, 그렇지 않은 위치에는 원래의 데이터를 사용
        masked_data = torch.where(mask.unsqueeze(-1), mask_token.expand_as(data), data)

        # 마스크를 int 형태로 변환하고 차원을 확장
        mask = mask.unsqueeze(-1).expand(-1, -1, feature_dim).int()

        return masked_data, rand_indices, mask



# if __name__ == '__main__':
#     # m = Decoder()
#     m = TSBendr()
#     d = torch.randn(size=(256,1,3000))
#     # target, pred, mask_idx, mask = m(d)
#     target, pred, mask = m(d)
#     aaa = m.evaluate(d)
#     # print(target.shape)
#     # print(pred.shape)
#     # print(mask.shape)
#     print(aaa.shape)
#
#     # m1 = SpectogramEncoder()
#     # m2 = SignalEncoder()
#     # d = torch.randn(size=(256, 1, 3000))  # B, C, sample(data)
#     # rep1 = m1(d)
#     # rep2 = m2(d)
#     # rep = rep1 + rep2
#     # print(rep.shape)
#
#     # target, pred, mask = m(d) # (256, 300, 192) -> (b,n,d) -> (batch, patch_num, feature_dim)
#     loss_reconstruct = F.l1_loss(target, pred, reduction='none')
#     loss = (loss_reconstruct * mask).sum() / (mask.sum() + 1e-5) # loss_recon * mask -> masking된 부분의 손실만 고려 -> mask의 sum 으로 나눠줌으로써 masking된 부분의 손실의 평균값을 구할수 있음
#     print(loss)
#     # d = torch.randn(size=(256,1,1,1500))
#     # m = PatchData()
#     # output = m(d)
#     # print(output)