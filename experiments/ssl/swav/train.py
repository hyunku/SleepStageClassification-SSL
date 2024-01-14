# -*- coding:utf-8 -*-
import ray
import mne
import sys
sys.path.append('../../../../EEG_Sebf_Supervised_Learning/experiments/ssl')

import torch
import random
import argparse
import settting
import warnings
import numpy as np
import torch.optim as opt
from experiments.ssl.swav.model import SWAV
from experiments.ssl.swav.data_loader import batch_dataloader
from settting import train_items, ft_items, eval_items

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

    # Train (for SWAV)
    parser.add_argument('--train_items', default=train_items, type=list)
    parser.add_argument('--train_epochs', default=500, type=int)
    parser.add_argument('--train_lr_rate', default=0.1, type=float)
    parser.add_argument('--train_batch_size', default=1024, type=int)
    parser.add_argument('--backbone_name', default='EEGNet', type=str,
                        choices=['EEGNet', 'ShallowConvNet'])
    parser.add_argument('--backbone_parameter', default={'f1': 8, 'f2': 16, 'd': 2,
                                                         'channel_size': len(settting.channels),
                                                         'input_time_length': settting.sampling_rate * settting.seconds,
                                                         'dropout_rate': 0.5,
                                                         'sampling_rate': settting.sampling_rate})

    # Setting Data Augmentation
    parser.add_argument('--augmentation_t1', default=['random_crop'])
    parser.add_argument('--augmentation_t2', default=['random_crop'])

    # Linear Evaluation & Fine-Tuning
    parser.add_argument('--ft_items', default=ft_items, type=list)
    parser.add_argument('--eval_items', default=eval_items, type=list)
    parser.add_argument('--labels', default=settting.labels)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = SWAV(backbone_name=self.args.backbone_name,
                          backbone_parameter=self.args.backbone_parameter)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)

    def train(self):
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(items=train_items, batch_size=self.args.train_batch_size,
                                            augmentation_t1=self.args.augmentation_t1,
                                            augmentation_t2=self.args.augmentation_t2)
        for epoch in range(self.args.train_epochs):
            for batch in train_dataloader.gather_async(num_async=5):
                x1, x2 = batch
                self.model((x1, x2))
                self.optimizer.zero_grad()
                self.optimizer.step()

        ray.shutdown()
