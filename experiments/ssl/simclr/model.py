# -*- coding:utf-8 -*-
import sys
sys.path.append('../../../../SSL_for_Sleep_EEG/experiments/ssl')

import torch.nn as nn
from models.utils import get_backbone_model, ProjectionHead


class SimCLR(nn.Module):
    def __init__(self, backbone_name, backbone_parameter, projection_hidden, projection_size):
        super().__init__()
        self.projection_hidden = projection_hidden
        self.projection_size = projection_size

        self.backbone = get_backbone_model(model_name=backbone_name,
                                           parameters=backbone_parameter)
        self.projector = ProjectionHead(
            in_features=self.backbone.final_length,
            hidden_features=self.projection_hidden,
            out_features=self.projection_size
        )
        self.fc = nn.Identity()

    def forward(self, x):
        x1, x2 = x
        feature_1 = self.backbone(x1)
        feature_2 = self.backbone(x2)
        out1 = self.projector(feature_1)
        out2 = self.projector(feature_2)
        return out1, out2


