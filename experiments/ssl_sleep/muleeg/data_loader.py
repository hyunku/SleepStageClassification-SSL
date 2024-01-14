# -*- coding:utf-8 -*-
import ray
import torch
import random
import numpy as np
from typing import List
from dataset.augmentation import SignalAugmentation as SigAug
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


def batch_dataloader(paths: List, batch_size: int, freqs: float):
    prob = 0.85
    weak_augmentation, strong_augmentation = [('random_crop', prob)], \
                                             [('random_permutation', prob), ('random_horizontal_flip', prob)]

    augmentation = SigAug(sampling_rate=freqs)
    np.random.shuffle(paths)

    def get_data(path: str) -> (np.array, np.array):
        data = np.load(path)['x']
        data = list(data)
        return data

    def convert_augmentation(x: List) -> (np.array, np.array):
        def converter(sample):
            if sample.ndim == 1:
                sample = np.expand_dims(sample, axis=0)

            # weak augmentation
            weak_sample, strong_sample = sample, sample
            for aug_name, aug_prob in weak_augmentation:
                weak_sample = augmentation.process(sample, aug_name=aug_name, p=aug_prob)

            # strong augmentation
            for aug_name, aug_prob in strong_augmentation:
                strong_sample = augmentation.process(sample, aug_name=aug_name, p=aug_prob)
            return weak_sample, strong_sample

        data = [converter(x_) for x_ in x]
        x1 = np.array([t[0] for t in data])
        x2 = np.array([t[1] for t in data])
        return x1, x2

    def convert_tensor(x1: np.array, x2: np.array) -> (torch.Tensor, torch.Tensor):
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        return x1, x2

    it = (
        ray.util.iter.from_items(paths, num_shards=5)
                     .for_each(lambda x_: get_data(x_))
                     .flatten()
                     .local_shuffle(shuffle_buffer_size=2)
                     .batch(batch_size)
                     .for_each(lambda x_: convert_augmentation(x_))
                     .for_each(lambda x_: convert_tensor(x_[0], x_[1]))
    )
    return it
