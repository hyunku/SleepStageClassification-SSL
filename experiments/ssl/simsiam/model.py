# -*- coding:utf-8 -*-
import torch.nn as nn
from models.utils import ProjectionHead
from models.backbone import get_backbone_model


class SimSiam(nn.Module):
    def __init__(self, backbone_name, backbone_parameter, projection_size, projection_hidden_size):
        super().__init__()
        self.encoder = AddProjHead(backbone_name=backbone_name,
                                   backbone_parameter=backbone_parameter,
                                   embedding_size=projection_size,
                                   hidden_size=projection_hidden_size)
        self.predictor = ProjectionHead(in_features=projection_size,
                                        hidden_features=projection_hidden_size,
                                        out_features=projection_size)

    def forward(self, x):
        x1, x2 = x

        # compute features for one view
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()


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
