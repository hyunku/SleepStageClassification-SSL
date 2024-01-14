# -*- coding:utf-8 -*-
import sys
sys.path.append('../../../../EEG_Sebf_Supervised_Learning/experiments/ssl')
import ray
import torch
import numpy as np
from typing import List
from dataset.augmentation import SignalAugmentation as SigAug
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def batch_dataloader(items: List, batch_size: int, augmentation_t1: List, augmentation_t2: List):
    np.random.shuffle(items)
    augmentation = SigAug()

    def get_data(paths_: str) -> np.array:
        data = np.load(paths_)
        x = data['x']
        return x

    def convert_augmentation(x) -> (np.array, np.array):
        x1, x2 = None, None
        for aug_t in augmentation_t1:
            x1 = augmentation.process(x, aug_name=aug_t)
        for aug_t in augmentation_t2:
            x2 = augmentation.process(x, aug_name=aug_t)
        return x1, x2

    def convert_tensor(x1, x2) -> (torch.Tensor, torch.Tensor):
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        return x1, x2

    it = (
        ray.util.iter.from_items(items, num_shards=5)
                     .for_each(lambda x_: get_data(x_))
                     .flatten()
                     .local_shuffle(shuffle_buffer_size=2)
                     .batch(batch_size)
                     .for_each(lambda x_: convert_augmentation(x_))
                     .for_each(lambda x_: convert_tensor(x_[0], x_[1]))
    )
    return it
