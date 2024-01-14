# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as f


class DiverseLoss(nn.Module):
    def __init__(self, temperature, intra_temperature):
        super(DiverseLoss, self).__init__()
        self.t = temperature
        self.intra_t = intra_temperature

    def loss(self, out_1: torch.Tensor, out_2: torch.Tensor):
        out_1 = f.normalize(out_1, p=2, dim=1)
        out_2 = f.normalize(out_2, p=2, dim=1)

        out = torch.cat([out_1, out_2], dim=0)  # 2B, 128
        n = out.shape[0]

        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous())  # 2B, 2B
        sim = torch.exp(cov / self.t)  # 2B, 2B

        # Negative similarity matrix
        mask = ~torch.eye(n, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n, -1).sum(dim=-1)

        # Positive similarity matrix
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.t)
        pos = torch.cat([pos, pos], dim=0)  # 2B
        loss = -torch.log(pos / neg).mean()
        return loss

    def intra_loss(self, weak_time: torch.Tensor, weak_spectral: torch.Tensor,
                   strong_time: torch.Tensor, strong_spectral: torch.Tensor) -> torch.Tensor:
        wt, ws = f.normalize(weak_time, p=2, dim=1), f.normalize(weak_spectral, p=2, dim=1)
        st, ss = f.normalize(strong_time, p=2, dim=1), f.normalize(strong_spectral, p=2, dim=1)

        out1 = torch.vstack((wt.unsqueeze(0), ws.unsqueeze(0)))
        out2 = torch.vstack((st.unsqueeze(0), ss.unsqueeze(0)))

        out = torch.cat([out1, out2], dim=0)  # 4*B*Feat
        n = out.shape[0]

        # similarity matrix
        cov = torch.einsum('abf,dbf->adb', out, out)  # /weak_time.shape[-1] # 4*4*B
        sim = torch.exp(cov / self.intra_t)

        # negative similarity matrix
        mask = ~torch.eye(n, device=sim.device).bool()
        neg = sim[mask].view(n, n - 1, weak_time.shape[0]).sum(dim=1)

        # positive similarity matrix
        pos = torch.exp(torch.sum(out1 * out2, dim=-1) / self.intra_t)
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg)
        loss = loss.mean()
        return loss

    def forward(self, w_t_feats, w_f_feats, w_s_feats, s_t_feats, s_f_feats, s_s_feats) -> torch.Tensor:
        l1 = self.loss(w_t_feats, s_t_feats)
        l2 = self.loss(w_f_feats, s_f_feats)
        l3 = self.loss(w_s_feats, s_s_feats)
        intra_loss = self.intra_loss(w_t_feats, w_s_feats, s_t_feats, s_s_feats)
        tot_loss = l1 + l2 + l3 + intra_loss
        return tot_loss
