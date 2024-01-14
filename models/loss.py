# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import sys

class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    @staticmethod
    def mask_correlated_samples(batch_size):
        n = 2 * batch_size
        mask = torch.ones((n, n), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_j.shape[0]
        n = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z = f.normalize(z, dim=-1)

        mask = self.mask_correlated_samples(batch_size)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n, 1)
        negative_samples = sim[mask].reshape(n, -1)

        labels = torch.from_numpy(np.array([0] * n)).reshape(-1).to(positive_samples.device).long()  # .float()
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature

        loss = self.criterion(logits, labels)
        loss /= n
        return loss, (labels, logits)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_i, z_j):
        # L2 Normalization
        z_i = f.normalize(z_i, dim=1)
        z_j = f.normalize(z_j, dim=1)
        return 2 - 2 * (z_i * z_j).sum(dim=-1)


class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, p1, p2, z1, z2):
        # Stop Gradient
        z1, z2 = z1.detach(), z2.detach()

        # L2 Normalization
        p1, p2 = f.normalize(p1, dim=1), f.normalize(p2, dim=1)
        z1, z2 = f.normalize(z1, dim=1), f.normalize(z2, dim=1)

        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss


class SWAVLoss(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, q_t, q_s, p_t, p_s):
        loss = -0.5 * (torch.mean(q_t * p_s) + torch.mean(q_s * p_t))
        return loss


class TripletLoss(nn.Module):
    def __init__(self, temperature, margin, sigma):
        super(TripletLoss, self).__init__()
        self.T = temperature
        self.margin = margin
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigma = sigma

    def forward(self, emb_anchor, emb_positive):
        # L2 normalize, Nxk, Nxk -> 1024, 256 -> B, K
        emb_anchor = f.normalize(emb_anchor, p=2, dim=1)
        emb_positive = f.normalize(emb_positive, p=2, dim=1)

        # compute instance-aware world representation, Nx1
        sim = torch.mm(emb_anchor, emb_positive.t()) / self.T # 두개 representation끼리 유사도 계산 (B, B)
        weight = self.softmax(sim) # 가중치 계산 (B, B)
        neg = torch.mm(weight, emb_positive) # (B,K) -> 이게 이해가 잘 안갑니다 ㅠㅠㅠㅠ

        # representation similarity of pos/neg pairs
        l_pos = torch.exp(-torch.sum(torch.pow(emb_anchor - emb_positive, 2), dim=1) / (2 * self.sigma ** 2))
        l_neg = torch.exp(-torch.sum(torch.pow(emb_anchor - neg, 2), dim=1) / (2 * self.sigma ** 2))

        zero_matrix = torch.zeros(l_pos.shape)
        loss = torch.max(zero_matrix, l_neg - l_pos + self.margin).mean()

        return loss

