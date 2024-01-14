# -*- coding:utf-8 -*-
import copy
import sys

import torch
import random
import numpy as np
import torch.nn as nn
from typing import List
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TorchDataset(Dataset):
    def __init__(self, paths: List):
        self.paths = paths
        self.xs, self.ys = self.get_data()

    def __len__(self):
        return self.xs.shape[0]

    def get_data(self):
        xs, ys = [], []
        for path in self.paths:
            data = np.load(path)
            x, y = data['x'], data['y']
            x = np.expand_dims(x, axis=1)
            xs.append(x)
            ys.append(y)
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        return xs, ys

    def __getitem__(self, idx):
        x = torch.tensor(self.xs[idx], dtype=torch.float)
        y = torch.tensor(self.ys[idx], dtype=torch.long)
        return x, y


class Model(nn.Module):
    def __init__(self, backbone: nn.Module, frozen_layers: List, classes=5):
        super().__init__()
        self.backbone = self.freeze_backbone(backbone, frozen_layers=frozen_layers)
        self.classes = classes
        self.fc = nn.Linear(self.backbone.emb_dim, self.classes)  # fc 레이어의 입력 차원을 SimMIM의 출력 차원과 일치하게 설정

    @staticmethod
    def freeze_backbone(backbone: nn.Module, frozen_layers: List):
        backbone = copy.deepcopy(backbone)
        for name, param in backbone.named_parameters():
            if name.split('.')[0] in frozen_layers:
                param.requires_grad = False
        return backbone

    def forward(self, x):
        x, _, _ = self.backbone(x)  # SimMIM의 forward 메소드가 3개의 값을 반환하므로, 첫 번째 값 사용
        x = self.fc(x)
        return x


class Evaluation(object):
    def __init__(self, backbone: nn.Module, device):
        self.backbone = backbone
        self.device = device
        self.epochs = 3 # TODO: check evaluation epochs -> default was 200
        self.lr = 0.01
        self.batch_size = 4096
        self.criterion = nn.CrossEntropyLoss()

    def evaluation(self, model: Model, eval_paths: List) -> (np.array, np.array):
        model.eval()

        total_pred, total_real = [], []
        for path in eval_paths:
            with torch.no_grad():
                data = np.load(path)
                x, y = data['x'], data['y']
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=1).to(self.device) # 3차원 데이터 -> 채널1로 하는 4차원 데이터로 변환 # TODO: BENDR에서 요류날지도??
                pred = model(x) # freeze 시킨 backbone을 사용하여 eval셋의 label 예측
                pred = pred.argmax(dim=-1).detach().cpu().numpy() # softmax 결과로 나온 확률값 matrix에서 최대값 뽑아내기 -> label 추출
                real = y # ground truth, 실제값
                total_pred.extend(pred) # iter한 객체들 추가
                total_real.extend(real)
        total_pred, total_real = np.array(total_pred), np.array(total_real)
        return total_pred, total_real

    def fine_tuning(self, ft_paths, eval_paths, frozen_layers: List) -> (np.array, np.array):
        # 1. [frozen = False] => Fine Tuning / 2. [frozen = True] => Backbone Frozen
        backbone = copy.deepcopy(self.backbone)
        model = Model(backbone=backbone, frozen_layers=frozen_layers).to(self.device) # 학습시킨 모델에서 특정 layer freeze
        model.train() # define finetune model

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr) # 모델에서 requires_grad = True 인 것의 가중치들만 옵티마이저에 등록
        train_dataset = TorchDataset(paths=ft_paths) # validation set
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        best_model, best_eval_mf1 = None, 0
        best_pred, best_real = None, None
        for epoch in range(self.epochs):
            model = self.train_mode(model, frozen_layers=frozen_layers)

            for data in train_dataloader:
                optimizer.zero_grad()

                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                optimizer.step()

            pred, real = self.evaluation(model=model, eval_paths=eval_paths)
            eval_acc, eval_mf1 = accuracy_score(y_true=real, y_pred=pred), \
                                 f1_score(y_true=real, y_pred=pred, average='macro')

            if eval_mf1 > best_eval_mf1:
                best_eval_mf1 = eval_mf1
                best_pred, best_real = pred, real

        del backbone, model, optimizer, train_dataset, train_dataloader
        return best_pred, best_real

    @staticmethod
    def train_mode(model, frozen_layers):
        for name, module in model.backbone.named_modules():
            if name in frozen_layers:
                module.eval()
        return model