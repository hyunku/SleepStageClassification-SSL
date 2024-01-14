# -*- coding:utf-8 -*-
import mne
import ray
import torch
import random
import argparse
import settting
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as opt
from experiments.evaluation import Evaluation
from torch.utils.data import dataloader, WeightedRandomSampler
from experiments.ssl.deepcluster.data_loader import *
from experiments.ssl.deepcluster.model import DeepCluster
from settting import train_items, ft_items, eval_items
from sklearn.metrics import classification_report


warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Setting Dataset
    parser.add_argument('--sampling_rate', default=settting.sampling_rate, type=float)
    parser.add_argument('--second', default=settting.seconds, type=int)

    # Train (for DeepCluster)
    parser.add_argument('--train_items', default=train_items, type=list)
    parser.add_argument('--train_epochs', default=500, type=int)
    parser.add_argument('--train_lr_rate', default=0.001, type=float)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--augmentations', default=[('random_bandpass_filter', 0.95),
                                                    ('random_crop', 0.95)])
    # Model
    parser.add_argument('--backbone_name', default='EEGNet', type=str,
                        choices=['EEGNet', 'ShallowConvNet'])
    parser.add_argument('--backbone_parameter', default={'f1': 8, 'f2': 16, 'd': 2,
                                                         'channel_size': len(settting.channels),
                                                         'input_time_length': settting.sampling_rate * settting.seconds,
                                                         'dropout_rate': 0.25,
                                                         'sampling_rate': settting.sampling_rate})
    parser.add_argument('--pca_n_components', default=128, type=int)
    parser.add_argument('--cluster_classes', default=len(settting.labels), type=int)

    # Evaluation & Fine-Tuning
    parser.add_argument('--ft_items', default=ft_items, type=list)
    parser.add_argument('--eval_items', default=eval_items, type=list)
    parser.add_argument('--labels', default=settting.labels)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = DeepCluster(backbone_name=self.args.backbone_name,
                                 backbone_parameter=self.args.backbone_parameter,
                                 pca_n_components=self.args.pca_n_components,
                                 cluster_classes=self.args.cluster_classes).to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)

    def generate_pseudo_labeling(self, data_loader):
        # Generator Pseudo Labeling
        xs, ys = [], []
        for x in data_loader.gather_async(num_async=5):
            x = x.to(device)
            try:
                # Assign pseudo-labels
                _, label = self.model(x, mode='pseudo_labels')
            except ValueError:  # for PCA Error
                continue
            xs.append(x)
            ys.append(label)
        xs, ys = torch.cat(xs, dim=0), torch.cat(ys, dim=0)
        return xs, ys

    def train(self):
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=1)
        train_dataloader = batch_dataloader(items=self.args.train_items,
                                            batch_size=256,
                                            augmentations=self.args.augmentations)

        self.compute_evaluation()

        train_count = 0
        for epoch in range(self.args.train_epochs):
            print('------------------------ [Train] ------------------------')
            xs, ys = self.generate_pseudo_labeling(data_loader=train_dataloader)

            # Init Set Fully Connected Layer
            self.model.projector = nn.Linear(in_features=self.model.backbone.final_length,
                                             out_features=self.args.cluster_classes)
            dataset = PseudoLabelDataset(xs=xs, ys=ys)
            data_loader = dataloader.DataLoader(dataset=dataset, batch_size=self.args.train_batch_size)
            optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
            self.model.train()

            for data in data_loader:
                optimizer.zero_grad()

                x, y = data
                out = self.model(x, mode='eval')
                loss = self.criterion(out, y)
                print('[Epoch] : {0:02d} \t [Iter] : {1:04d} \t [Loss] : {2:.3f} \t [Acc]: {3:.4f}'.format(
                    epoch + 1, train_count, loss.item(), self.compute_metrics(out, y)))

                train_count += 1
                loss.backward()
                optimizer.step()

            self.compute_evaluation()
        ray.shutdown()

    def compute_evaluation(self):
        print('------------------------ [Evaluation] ------------------------')
        evaluation = Evaluation(backbone=self.model.backbone, labels=self.args.labels, device=device)
        test_pred, test_real = evaluation.svm(train_items=self.args.ft_items,
                                              test_items=self.args.eval_items)
        print(classification_report(y_true=test_real, y_pred=test_pred))

    @staticmethod
    def compute_metrics(output, target):
        output = output.argmax(dim=-1)
        accuracy = torch.mean(torch.eq(target, output).to(torch.float32))
        return accuracy


if __name__ == '__main__':
    augments = get_args()
    trainer = Trainer(augments)
    trainer.train()
