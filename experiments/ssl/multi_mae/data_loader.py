# -*- coding:utf-8 -*-
import ray
import torch
import random
import numpy as np
from typing import List, Dict
from collections import OrderedDict
from dataset.utils import butter_bandpass_filter
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


def batch_dataloader(paths: List, band_range: OrderedDict, sampling_rate: int, batch_size: int):
    np.random.shuffle(paths)

    def get_data(path: str) -> (np.array, np.array):
        data = np.load(path)['x']
        data = list(data)
        return data

    def convert_data(x: List) -> np.array:
        def converter(sample):
            total_s = []
            for b_name, b_range in band_range.items():
                low_cut, high_cut = b_range
                s = butter_bandpass_filter(sample, low_cut, high_cut, fs=sampling_rate)
                total_s.append(s)
            total_s = np.array(total_s)
            return total_s

        data = np.array([converter(x_) for x_ in x])
        return data

    def convert_tensor(x: np.array) -> Dict:
        new_x = {}
        for i, b_name in enumerate(band_range.keys()):
            new_x[b_name] = torch.unsqueeze(
                torch.tensor(x[:, i, :], dtype=torch.float32),
                dim=1)
        return new_x

    it = (
        ray.util.iter.from_items(paths, num_shards=5)
                     .for_each(lambda x_: get_data(x_))
                     .flatten()
                     .local_shuffle(shuffle_buffer_size=2)
                     .batch(batch_size)
                     .for_each(lambda x_: convert_data(x_))
                     .for_each(lambda x_: convert_tensor(x_))
    )
    return it
