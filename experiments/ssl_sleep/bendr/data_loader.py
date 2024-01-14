# -*- coding:utf-8 -*-
import ray
import torch
import random
import numpy as np
from typing import List
import warnings
from dataset.utils import butter_bandpass_filter

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 424
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

# TODO: Original - 30초

# def batch_dataloader(paths: List, batch_size: int, seq_size: int):
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



# TODO: 60초 단위 데이터
# def batch_dataloader(paths: List, batch_size: int, seq_size: int):
#     np.random.shuffle(paths)
#
#     def get_data(path: str) -> (np.array, np.array):
#         data = np.load(path)['x']
#         return list(data)
#
#     def convert_tensor(x: np.array) -> torch.Tensor:
#         x = torch.tensor(x, dtype=torch.float32)
#         x = torch.unsqueeze(x, dim=1)
#         return x
#
#     def concatenate_data(data_list):
#         concatenated_data = []
#         for i in range(0, len(data_list), 2):
#             if i+1 < len(data_list):
#                 concatenated_data.append(np.concatenate((data_list[i], data_list[i+1]), axis=0))
#         return concatenated_data
#
#     it = (
#         ray.util.iter.from_items(paths, num_shards=5)
#                      .for_each(lambda x_: get_data(x_))
#                      .flatten()
#                      .batch(2)  # 2개씩 묶어서 6000으로 만듬
#                      .for_each(concatenate_data)
#                      .local_shuffle(shuffle_buffer_size=2)
#                      .batch(batch_size)
#                      .for_each(lambda x_: convert_tensor(x_))
#     )
#     return it


# TODO: 60초 이상 데이터
# def batch_dataloader(paths: List, batch_size: int, seq_size: int):
#     np.random.shuffle(paths)
#
#     def get_data(path: str) -> (np.array, np.array):
#         data = np.load(path)['x']
#         return list(data)
#
#     def convert_tensor(data_list: List[np.array]) -> torch.Tensor:
#         return torch.tensor(data_list, dtype=torch.float32)
#
#     chunk_size = 3000
#     chunks_needed = seq_size // chunk_size
#
#     def concatenate_data(data_list):
#         concatenated_data = []
#
#         total_chunks = len(data_list)
#
#         if total_chunks < chunks_needed:
#             return []
#
#         for i in range(0, total_chunks - chunks_needed + 1):
#             temp_data = data_list[i:i + chunks_needed]
#             concatenated_chunk = np.concatenate(temp_data, axis=0)
#             if concatenated_chunk.shape[0] == seq_size:
#                 concatenated_data.append(concatenated_chunk)
#
#         return concatenated_data
#
#     it = (
#         ray.util.iter.from_items(paths, num_shards=5)
#         .for_each(lambda x_: get_data(x_))
#         .flatten()
#         .batch(chunks_needed)
#         .for_each(concatenate_data)
#         .filter(lambda x: len(x) > 0)
#         .local_shuffle(shuffle_buffer_size=2)
#         .batch(batch_size)
#         .for_each(lambda x_: convert_tensor(x_))
#     )
#     return it


# TODO: 30초 단위 + normalize + bandpass filtering
def compute_global_mean_std(paths: List[str]) -> (float, float):
    all_data = []
    for path in paths:
        data = np.load(path)['x']
        data = butter_bandpass_filter(data, low_cut=1, high_cut=40, fs=100)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    mean = np.mean(all_data)
    std = np.std(all_data)
    return mean, std

def batch_dataloader(paths: List, batch_size: int, seq_size: int):
    np.random.shuffle(paths)
    mean, std = compute_global_mean_std(paths)

    def get_data(path: str) -> np.array:
        data = np.load(path)['x']
        data = butter_bandpass_filter(data, low_cut=1, high_cut=40, fs=100)
        data = (data - mean) / std
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




