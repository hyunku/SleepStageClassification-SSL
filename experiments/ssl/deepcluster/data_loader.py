# -*- coding:utf-8 -*-
import sys
sys.path.append('../../../../EEG_Self_Supervised_Learning/experiments/ssl')

import ray
import torch
import random
import numpy as np
from typing import List
from torch.utils.data import Dataset
from dataset.augmentation import SignalAugmentation as SigAug
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def batch_dataloader(items: List, batch_size: int, augmentations: List):
    np.random.shuffle(items)
    augmentation = SigAug()

    def get_data(paths_: str) -> np.array:
        data = np.load(paths_)
        x = data['x']
        return x

    def convert_augmentation(x) -> np.array:
        aug_name, aug_prob = random.sample(augmentations, 1)[0]
        x = augmentation.process(x, aug_name=aug_name, p=aug_prob)
        return x

    def convert_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
        return x

    it = (
        ray.util.iter.from_items(items, num_shards=5)
                     .for_each(lambda x_: get_data(x_))
                     .flatten()
                     .local_shuffle(shuffle_buffer_size=2)
                     .batch(batch_size)
                     .for_each(lambda x_: convert_augmentation(x_))
                     .for_each(lambda x_: convert_tensor(x_))
    )
    return it


class PseudoLabelDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x, y = self.xs[idx], self.ys[idx]
        x, y = x.to(torch.float32), y.to(torch.long)
        return x, y
