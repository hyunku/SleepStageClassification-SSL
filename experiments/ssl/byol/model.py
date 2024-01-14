# -*- coding:utf-8 -*-
import sys
sys.path.append('../../../../SSL_BCI_EEG/experiments/ssl')
import torch
import copy
import torch.nn as nn
from models.utils import get_backbone_model, ProjectionHead


class BYOL(nn.Module):
    def __init__(self, backbone_name, backbone_parameter, projection_size, projection_hidden_size,
                 moving_average_decay=0.99, use_momentum=True):
        super().__init__()
        self.student_model = AddProjHead(backbone_name=backbone_name,
                                         backbone_parameter=backbone_parameter,
                                         hidden_size=projection_hidden_size,
                                         embedding_size=projection_size)
        self.teacher_model = self._get_teacher()
        self.target_ema_updater = EMA(moving_average_decay)
        self.student_predictor = ProjectionHead(in_features=projection_size,
                                                hidden_features=projection_hidden_size,
                                                out_features=projection_size)
        self.use_momentum = use_momentum
        self.fc = nn.Identity()

    @torch.no_grad()
    def _get_teacher(self):
        return copy.deepcopy(self.student_model)

    @torch.no_grad()
    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum ' \
                                  'for the target encoder '
        assert self.teacher_model is not None, 'target encoder has not been created yet'

        for student_params, teacher_params in zip(self.student_model.parameters(), self.teacher_model.parameters()):
            old_weight, up_weight = teacher_params.data, student_params.data
            teacher_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def forward(self, x):
        x1, x2 = x
        # student projection: backbone + MLP projection
        student_proj_one = self.student_model(x1)
        student_proj_two = self.student_model(x2)

        # additional student's MLP head called predictor
        student_pred_one = self.student_predictor(student_proj_one)
        student_pred_two = self.student_predictor(student_proj_two)

        with torch.no_grad():
            # teacher processes the images and makes projections: backbone + MLP
            teacher_proj_two = self.teacher_model(x1).detach_()
            teacher_proj_one = self.teacher_model(x2).detach_()

        return {
            'x1': {'student': student_pred_one, 'teacher': teacher_proj_one},
            'x2': {'student': student_pred_two, 'teacher': teacher_proj_two},
        }


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


class EMA(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.alpha + (1 - self.alpha) * new

