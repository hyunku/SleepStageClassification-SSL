# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
from models.utils import get_backbone_model, ProjectionHead
import torch.nn.functional as f
import sys


# Tip: 기존에 있는 모델 마지막 fc만 가로채서 수정 => conv_final_length : self.encoder.fc.weight.shape[1]

class MoCo(nn.Module): # Ver3, q: base encoder + additional fc(predictor), k: momentum encoder + not use queue
    def __init__(self, backbone_name, backbone_parameter, projection_hidden, projection_size, temperature, queue_dim):
        super().__init__()
        self.projection_hidden = projection_hidden
        self.projection_size = projection_size

        self.encoder_q = get_backbone_model(model_name=backbone_name,
                                            parameters=backbone_parameter)
        self.encoder_k = get_backbone_model(model_name=backbone_name,
                                            parameters=backbone_parameter)
        self.projector_q = ProjectionHead(in_features=self.encoder_q.final_length,
                                          hidden_features=self.projection_hidden,
                                          out_features=self.projection_size,
                                          head_type='nonlinear')
        self.projector_k = ProjectionHead(in_features=self.encoder_k.final_length,
                                          hidden_features=self.projection_hidden,
                                          out_features=self.projection_size,
                                          head_type='nonlinear')
        self.predictor_q = ProjectionHead(in_features=self.projection_size,
                                          hidden_features=self.projection_hidden,
                                          out_features=self.projection_size,
                                          head_type='nonlinear')
        self.T = temperature
        self.m = 0.999

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize -> 인코더값(encoder_q)을 momentum encoder(encoder_k) 에 copy 해주는 것
            param_k.requires_grad = False  # not update by gradient -> momentum encoder는 학습하지 않음.

    def forward(self, x, mode='train'):
        if mode == 'train':
            x1, x2 = x # x1, x2 shape : (256, 1, 3000) # (batch, height(eeg chan), width(time))
            q1 = self.predictor_q(self.projector_q(self.encoder_q(x1)))
            q2 = self.predictor_q(self.projector_q(self.encoder_q(x2)))

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                k1 = self.projector_k(self.encoder_k(x1))
                k2 = self.projector_k(self.encoder_k(x2))

            return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


        elif mode == 'eval':
            feature = self.backbone(x)
            out = self.fc(feature)
            return out

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m) # momentum update


    def contrastive_loss(self, q, k):
        # normalize
        q = f.normalize(q, dim=1)
        k = f.normalize(k, dim=1)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        loss = nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
        return loss
