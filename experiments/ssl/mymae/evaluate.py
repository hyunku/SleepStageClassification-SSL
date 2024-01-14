# -*- coding:utf-8 -*-
import copy
import torch
import random
import numpy as np
import torch.nn as nn
from typing import List
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


random_seed = 424
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
        self.backbone = self.get_backbone(backbone=backbone, frozen_layers=frozen_layers)
        self.classes = classes
        self.fc = nn.Linear(backbone.final_length, self.classes)

    @staticmethod
    def get_backbone(backbone: nn.Module, frozen_layers: List):
        backbone = copy.deepcopy(backbone)
        for name, param in backbone.named_parameters():
            if name in frozen_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
        return backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class Evaluation(object):
    def __init__(self, backbone: nn.Module, device):
        self.backbone = backbone
        self.device = device
        self.epochs = 50
        self.lr = 0.0005
        self.batch_size = 256
        self.criterion = nn.CrossEntropyLoss()

    def evaluation(self, model: Model, eval_paths: List) -> (np.array, np.array):
        model.eval()
        total_pred, total_real = [], []
        for path in eval_paths:
            with torch.no_grad():
                data = np.load(path)
                x, y = data['x'], data['y'] # (data_num(b), data(3000))
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=1).unsqueeze(dim=1).to(self.device) # (b,1,1,3000)
                pred = model(x)
                pred = pred.argmax(dim=-1).detach().cpu().numpy()
                real = y
                total_pred.extend(pred)
                total_real.extend(real)
        total_pred, total_real = np.array(total_pred), np.array(total_real)
        return total_pred, total_real

    def fine_tuning(self, ft_paths, eval_paths, frozen_layers: List) -> (np.array, np.array):
        # 1. [frozen = False] => Fine Tuning / 2. [frozen = True] => Backbone Frozen
        backbone = copy.deepcopy(self.backbone) # encoder

        model = Model(backbone=backbone, frozen_layers=frozen_layers).to(self.device)
        # for name, parameter in model.named_parameters():
        #     print(f"[ Name ] : {name} [ Parameter ] : {parameter.requires_grad}")

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        train_dataset = TorchDataset(paths=ft_paths)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        best_model, best_eval_mf1 = None, 0
        best_pred, best_real = None, None
        for epoch in range(self.epochs):
            model.train()

            for data in train_dataloader:
                optimizer.zero_grad()

                x, y = data
                x = x.unsqueeze(1) # x : (b,1,1,3000)
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x) # pred : (b,5)
                loss = self.criterion(pred, y)

                loss.backward()
                optimizer.step()

            pred, real = self.evaluation(model=model, eval_paths=eval_paths)
            eval_acc, eval_mf1 = accuracy_score(y_true=real, y_pred=pred), \
                                 f1_score(y_true=real, y_pred=pred, average='macro')
            # print(pred, end='\t')
            # print(eval_mf1)

            if eval_mf1 > best_eval_mf1:
                best_eval_mf1 = eval_mf1
                best_pred, best_real = pred, real

        del backbone, model, optimizer, train_dataset, train_dataloader
        return best_pred, best_real

    @staticmethod
    def train_mode(model, frozen_layers: List):
        for name, module in model.named_modules():
            if name in frozen_layers:
                module.eval()
            else:
                module.train()
        return model
