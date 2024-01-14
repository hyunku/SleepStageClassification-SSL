# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
from models.utils import get_backbone_model
from models.utils import get_backbone_parameter
import torch.nn.functional as f
import sys
import copy
import numpy as np


class BENDR(nn.Module):
    def __init__(self, backbone_name, backbone_parameter, mask_span, mask_rate, num_negatives, temperature,
                 context_dim, context_heads, context_layers, context_dropouts):
        super().__init__()
        self.encoder = get_backbone_model(model_name=backbone_name,
                                          parameters=backbone_parameter)
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temperature
        self.num_negatives = num_negatives
        self.contextualizer = Contextualizer(in_features=256, hidden_feedforward=context_dim, heads=context_heads, layers=context_layers,
                                             dropout=context_dropouts)  # hid, drop is for transformerencoder parameters

    def _generate_negatives(self, z):
        """Generate negative samples to compare each sequence location against"""
        batch_size, feat, full_len = z.shape
        z_k = z.permute([0, 2, 1]).reshape(-1, feat)
        with torch.no_grad():
            # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
            negative_inds = torch.randint(0, full_len-1, size=(batch_size, full_len * self.num_negatives))
            # From wav2vec 2.0 implementation, I don't understand
            # negative_inds[negative_inds >= candidates] += 1

            for i in range(1, batch_size):
                negative_inds[i] += i * full_len # negative idx 선택

        z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.num_negatives, feat) # z_k에서 선택된 negative idx들로부터 negative vector 가져오고, 형태 맞춰줌
        return z_k, negative_inds # negative sample들의 feature vector, idx들

    def _calculate_similarity(self, z, c, negatives):
        c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
        z = z.permute([0, 2, 1]).unsqueeze(-2)

        # In case the contextualizer matches exactly, need to avoid divide by zero errors
        negative_in_target = (c == negatives).all(-1)
        targets = torch.cat([c, negatives], dim=-2)

        logits = f.cosine_similarity(z, targets, dim=-1) / self.temp
        if negative_in_target.any():
            logits[1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])


    def forward(self, x: torch.Tensor):
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[-1]))
        x = self.encoder(x) # b, feat, seq -> 64, 512, 32
        unmasked_x = x.clone()
        b, feat, samples = x.shape
        mask = _make_mask((b, samples), self.mask_rate, samples, self.mask_span)
        c = self.contextualizer(x, mask) # b, feat, seq -> 64, 512, 32 masking하면 33

        # Select negative candidates and generate labels for which are correct labels
        negatives, negative_inds = self._generate_negatives(x)

        # Prediction -> batch_size x predict_length x predict_length
        logits = self._calculate_similarity(unmasked_x, c, negatives)

        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return logits, labels, mask



class Permute(torch.nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


class Contextualizer(torch.nn.Module):
    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8,
                 dropout=0.15, activation='gelu', position_encoder=25, start_token=-5):
        super().__init__()
        self.dropout = dropout
        self.start_token = start_token
        self.in_features = in_features
        self._transformer_dim = in_features * 3
        encoder = torch.nn.TransformerEncoderLayer(d_model=in_features * 3,
                                                   nhead=heads, dim_feedforward=hidden_feedforward,
                                                   dropout=dropout, activation=activation)

        self.norm = torch.nn.LayerNorm(self._transformer_dim)
        self.transformer_layers = torch.nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)))
        self.position_encoder = position_encoder > 0 # True
        if position_encoder:
            conv = torch.nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            torch.nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            torch.nn.init.constant_(conv.bias, 0)
            conv = torch.nn.utils.weight_norm(conv, dim=2)
            self.relative_position = torch.nn.Sequential(conv, torch.nn.GELU())
        self.input_conditioning = torch.nn.Sequential(
            Permute([0, 2, 1]),
            torch.nn.LayerNorm(in_features),
            torch.nn.Dropout(dropout),
            Permute([0, 2, 1]),
            torch.nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )
        self.output_layer = torch.nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data


    def forward(self, x, mask_t=None):
        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))



def _make_mask(shape, p, total, span, allow_no_inds=False):
    # num_mask_spans = np.sum(np.random.rand(total) < p)
    # num_mask_spans = int(p * total)
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask

def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)




if __name__ == '__main__':
    m = BENDR(
        backbone_name='BENDREncoder',
        backbone_parameter=get_backbone_parameter(model_name='BENDREncoder', sampling_rate=125),
        mask_span=6,
        mask_rate=0.1,
        num_negatives=100,
        temperature=0.5)
    d = torch.randn(size=(64, 1, 3000)) # B, C, sample(data)
    logits, x, mask = m(d)
    print(x.shape)
