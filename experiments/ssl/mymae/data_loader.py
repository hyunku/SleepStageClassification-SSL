# -*- coding:utf-8 -*-
import ray
import torch
import random
import numpy as np
from typing import List
import warnings
from dataset.utils import butter_bandpass_filter
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 424
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


# TODO: Original
# def batch_dataloader(paths: List, batch_size: int):
#     np.random.shuffle(paths)
#
#     def get_data(path: str) -> (np.array, np.array):
#         data = np.load(path)['x']
#         data = list(data)
#         return data
#
#     def convert_tensor(x: np.array) -> torch.Tensor:
#         x = torch.tensor(x, dtype=torch.float32)
#         x = torch.unsqueeze(x, dim=1)
#         x = torch.unsqueeze(x, dim=1)
#         return x
#
#     it = (
#         ray.util.iter.from_items(paths, num_shards=5)
#                      .for_each(lambda x_: get_data(x_))
#                      .flatten()
#                      .local_shuffle(shuffle_buffer_size=2)
#                      .batch(batch_size)
#                      .for_each(lambda x_: convert_tensor(x_))
#     )
#     return it


# TODO: filtering + std
def compute_global_mean_std(paths: List[str]) -> (float, float):
    all_data = []
    for path in paths:
        data = np.load(path)['x']
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_data = butter_bandpass_filter(all_data, low_cut=0.5, high_cut=40, fs=100, order=5)
    mean = np.mean(all_data)
    std = np.std(all_data)
    return mean, std, all_data

def batch_dataloader(paths: List, batch_size: int):
    mean, std, all_filtered_data = compute_global_mean_std(paths)

    def get_data(batch_data: np.array) -> np.array:
        batch_data = (batch_data - mean) / std  # standard scaling
        batch_data = list(batch_data)
        return batch_data

    def convert_tensor(x: np.array) -> torch.Tensor:
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.unsqueeze(x, dim=1)
        x = torch.unsqueeze(x, dim=1)
        return x

    num_batches = len(all_filtered_data) // batch_size
    it = (
        ray.util.iter.from_items([all_filtered_data[i*batch_size:(i+1)*batch_size] for i in range(num_batches)], num_shards=5)
                     .for_each(lambda x_: get_data(x_))
                     .for_each(lambda x_: convert_tensor(x_))
    )
    return it



# TODO: TorchDataSet Ver.
# class FilteredDataset(Dataset):
#     def __init__(self, paths):
#         all_data = []
#         for path in paths:
#             data = np.load(path)['x']
#             all_data.append(data)
#         all_data = np.concatenate(all_data, axis=0)
#
#         self.data = butter_bandpass_filter(all_data, low_cut=1, high_cut=40, fs=100, order=5)
#         self.mean = np.mean(self.data)
#         self.std = np.std(self.data)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         sample = (sample - self.mean) / self.std
#         sample = torch.tensor(sample, dtype=torch.float32)
#         sample = torch.unsqueeze(sample, dim=0)
#         return sample
#
# def batch_dataloader(paths: List, batch_size: int):
#     dataset = FilteredDataset(paths)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return dataloader