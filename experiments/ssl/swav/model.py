# -*- coding:utf-8 -*-
import sys
sys.path.append('../../../../EEG_Sebf_Supervised_Learning/experiments/ssl')
import torch
import torch.nn as nn
from models.utils import get_backbone_model


class SWAV(nn.Module):
    def __init__(self, backbone_name, backbone_parameter):
        super().__init__()
        self.backbone = get_backbone_model(model_name=backbone_name,
                                           parameters=backbone_parameter)

    def forward(self, x):
        x1, x2 = x
        feature1 = self.backbone(x1)
        feature2 = self.backbone(x2)

        return x

