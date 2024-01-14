# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/brainlab/Workspace/HG/SleepEEG/')
sys.path.append('/home/brainlab/Workspace/HG/SleepEEG/ssl/')
sys.path.append('/home/brainlab/Workspace/HG/SleepEEG/experiments/')
sys.path.append('/home/brainlab/Workspace/HG/SleepEEG/ssl/simmim/')

import ray
import torch
import random
import numpy as np
from typing import List
import warnings


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


def batch_dataloader(paths: List, batch_size: int):
    np.random.shuffle(paths)

    def get_data(path: str) -> (np.array, np.array):
        data = np.load(path)['x']
        data = list(data)
        return data

    def convert_tensor(x: np.array) -> torch.Tensor:
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.unsqueeze(x, dim=1)
        x = torch.unsqueeze(x, dim=1)
        return x

    it = (
        ray.util.iter.from_items(paths, num_shards=5)
                     .for_each(lambda x_: get_data(x_))
                     .flatten()
                     .local_shuffle(shuffle_buffer_size=2)
                     .batch(batch_size)
                     .for_each(lambda x_: convert_tensor(x_))
    )
    return it
