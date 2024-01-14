# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
from models.utils import get_backbone_model, ProjectionHead


class ContraWR(nn.Module):
    def __init__(self, backbone_name, backbone_parameter, projection_hidden, projection_size, ema_momentum):
        super().__init__()
        self.projection_hidden = projection_hidden
        self.projection_size = projection_size
        self.ema_momentum = ema_momentum

        self.encoder_q = AddProjHead(backbone_name=backbone_name,
                                     backbone_parameter=backbone_parameter,
                                     embedding_size=projection_size,
                                     hidden_size=projection_hidden)
        self.encoder_k = AddProjHead(backbone_name=backbone_name,
                                     backbone_parameter=backbone_parameter,
                                     embedding_size=projection_size,
                                     hidden_size=projection_hidden)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # not update by gradient

    def forward(self, x):
        x1, x2 = x
        out1 = self.encoder_q(x1)

        with torch.no_grad():
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.ema_momentum + param_q.data * (1. - self.ema_momentum)
            out2 = self.encoder_k(x2)

        return out1, out2


class AddProjHead(nn.Module):
    def __init__(self, backbone_name, backbone_parameter, embedding_size, hidden_size):
        super().__init__()
        self.backbone = get_backbone_model(model_name=backbone_name,
                                           parameters=backbone_parameter)
        self.projection = ProjectionHead(in_features=self.backbone.final_length,
                                         hidden_features=hidden_size,
                                         out_features=embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x
