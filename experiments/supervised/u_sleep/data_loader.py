# -*- coding:utf-8 -*-
import ray
import torch
import random
import numpy as np
from typing import List
import warnings
from dataset.utils import butter_bandpass_filter
from scipy.signal import resample
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 424
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


# TODO: Robust Scaling -> Paper
# class TorchDataset(Dataset):
#     def __init__(self, paths: List):
#         self.paths = paths
#         self.xs, self.ys = self.get_data()
#
#     def __len__(self):
#         return self.xs.shape[0]
#
#     def get_data(self):
#         xs, ys = [], []
#         for path in self.paths:
#             data = np.load(path)
#             x, y = data['x'], data['y']
#
#             # preprocess -> robust scaling
#             for i in range(x.shape[1]): # for each chan
#                 channel_data = x[:, i]
#                 median = np.median(channel_data)
#                 iqr = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)
#                 channel_data = (channel_data - median) / iqr # scale median to 0, iqr to 1
#                 channel_data = np.clip(channel_data, -20*iqr, 20*iqr) # IQR 20배 이상 차이나는 곳 클리핑
#
#                 x[:, i] = channel_data
#
#             x = np.expand_dims(x, axis=1)
#             x = np.expand_dims(x, axis=1)
#             xs.append(x)
#             ys.append(y)
#         xs = np.concatenate(xs, axis=0)
#         ys = np.concatenate(ys, axis=0)
#         return xs, ys
#
#     def __getitem__(self, idx):
#         x = torch.tensor(self.xs[idx], dtype=torch.float)
#         y = torch.tensor(self.ys[idx], dtype=torch.long)
#         return x, y
#
# def batch_dataloader(paths: List, batch_size: int, shuffle: bool):
#     dataset = TorchDataset(paths)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader


# TODO: Non-Preprocess
# def batch_dataloader(paths: List, batch_size: int):
#     np.random.shuffle(paths)
#
#     def get_data(path: str) -> (torch.Tensor, torch.Tensor):
#         data = np.load(path)
#         x_data = data['x']
#         y_data = data['y']
#         return x_data, y_data
#
#     def convert_tensor(x: np.array, y: np.array) -> (torch.Tensor, torch.Tensor):
#         x_tensor = torch.tensor(x, dtype=torch.float32)
#         x_tensor = torch.unsqueeze(x_tensor, dim=1)
#         x_tensor = torch.unsqueeze(x_tensor, dim=1)
#
#         y_tensor = torch.tensor(y, dtype=torch.long)
#         return x_tensor, y_tensor
#
#     it = (
#         ray.util.iter.from_items(paths, num_shards=5)
#                      .for_each(lambda x_: get_data(x_))
#                      .flatten()
#                      .local_shuffle(shuffle_buffer_size=2)
#                      .batch(batch_size)
#                      .for_each(lambda batch: [convert_tensor(x, y) for x, y in zip(*batch)])
#     )
#     return it

# TODO: Filtering + std scaling
class FilteredDataset(Dataset):
    def __init__(self, paths):
        all_x_data = []
        all_y_data = []
        for path in paths:
            data = np.load(path)
            x, y = data['x'], data['y']
            all_x_data.append(x)
            all_y_data.append(y)
        all_x_data = np.concatenate(all_x_data, axis=0)
        all_y_data = np.concatenate(all_y_data, axis=0)

        self.x_data = butter_bandpass_filter(all_x_data, low_cut=1, high_cut=40, fs=100, order=5)
        self.y_data = all_y_data
        self.mean = np.mean(self.x_data)
        self.std = np.std(self.x_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_sample = self.x_data[idx]
        x_sample = (x_sample - self.mean) / self.std
        x_sample = torch.tensor(x_sample, dtype=torch.float32)
        x_sample = torch.unsqueeze(x_sample, dim=0)

        y_sample = self.y_data[idx]
        y_sample = torch.tensor(y_sample, dtype=torch.long)

        return x_sample, y_sample

def batch_dataloader(paths: List, batch_size: int, shuffle: bool):
    dataset = FilteredDataset(paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# TODO: 35 segment + std + filtering
# class FilteredDataset(Dataset):
#     def __init__(self, paths, segment_size=35):
#         self.segment_size = segment_size
#         all_x_data = []
#         all_y_data = []
#
#         for path in paths:
#             data = np.load(path)
#             x, y = data['x'], data['y']
#
#             # filtering
#             filtered_x = butter_bandpass_filter(x, low_cut=1, high_cut=40, fs=100, order=5)
#
#             num_segments = len(filtered_x) // segment_size
#
#             # make segment
#             for i in range(num_segments):
#                 start = i * segment_size
#                 end = start + segment_size
#                 all_x_data.append(filtered_x[start:end].reshape(-1))
#                 all_y_data.append(y[start:end])
#
#         self.x_data = np.array(all_x_data).reshape(-1, 1, 3000 * segment_size)
#         self.y_data = np.array(all_y_data)
#
#         self.mean = np.mean(self.x_data)
#         self.std = np.std(self.x_data)
#
#     def __len__(self):
#         return len(self.x_data)
#
#     def __getitem__(self, idx):
#         x_sample = self.x_data[idx]
#         x_sample = (x_sample - self.mean) / self.std # standard scaling
#         x_sample = torch.tensor(x_sample, dtype=torch.float32)
#
#         y_sample = self.y_data[idx]
#         y_sample = torch.tensor(y_sample, dtype=torch.long)
#
#         return x_sample, y_sample
#
# def batch_dataloader(paths: List, batch_size: int, shuffle: bool, segment_size: int):
#     dataset = FilteredDataset(paths=paths, segment_size=segment_size)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader


# TODO: 35 segment + robust scaling(paper)
# class FilteredDataset(Dataset):
#     def __init__(self, paths, segment_size=35):
#         self.segment_size = segment_size
#         all_x_data = []
#         all_y_data = []
#
#         for path in paths:
#             data = np.load(path)
#             x, y = data['x'], data['y']
#
#             for i in range(x.shape[1]): # for each chan
#                 channel_data = x[:, i]
#                 median = np.median(channel_data)
#                 iqr = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)
#                 channel_data = (channel_data - median) / iqr # scale median to 0, iqr to 1
#                 channel_data = np.clip(channel_data, -20*iqr, 20*iqr) # IQR 20배 이상 차이나는 곳 클리핑
#
#                 x[:, i] = channel_data
#
#             num_segments = len(x) // segment_size
#
#             # make segment
#             for i in range(num_segments):
#                 start = i * segment_size
#                 end = start + segment_size
#                 all_x_data.append(x[start:end].reshape(-1))
#                 all_y_data.append(y[start:end])
#
#         self.x_data = np.array(all_x_data).reshape(-1, 1, 3000 * segment_size)
#         self.y_data = np.array(all_y_data)
#
#     def __len__(self):
#         return len(self.x_data)
#
#     def __getitem__(self, idx):
#         x_sample = self.x_data[idx]
#         x_sample = torch.tensor(x_sample, dtype=torch.float32)
#
#         y_sample = self.y_data[idx]
#         y_sample = torch.tensor(y_sample, dtype=torch.long)
#
#         return x_sample, y_sample
#
# def batch_dataloader(paths: List, batch_size: int, shuffle: bool, segment_size: int):
#     dataset = FilteredDataset(paths=paths, segment_size=segment_size)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader



