import torch
import random
import mne
import numpy as np
from torch.utils.data import Dataset
from dataset.utils import butter_bandpass_filter
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


class TorchDataset(Dataset):
    def __init__(self, paths, temporal_context_length, window_size, sampling_rate: int = 100):
        super().__init__()
        self.x, self.y = self.get_data(paths, sampling_rate)
        self.x, self.y = self.many_to_many(self.x,
                                           temporal_context_length=temporal_context_length,
                                           window_size=window_size), \
                         self.many_to_many(self.y, temporal_context_length=temporal_context_length,
                                           window_size=window_size)
        self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

    @staticmethod
    def get_data(paths, sampling_rate):
        total_x, total_y = [], []
        for path in paths:
            data = np.load(path)
            x, y = data['x'], data['y']

            # RobustScaler
            scaler = mne.decoding.Scaler(info=mne.create_info(ch_names=['Fpz'],
                                                              sfreq=sampling_rate,
                                                              ch_types='eeg'),
                                         scalings='median')
            x = np.expand_dims(x, axis=1)
            x = scaler.fit_transform(x)
            total_x.append(x.squeeze())
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
        return total_x, total_y

    @staticmethod
    def many_to_many(elements, temporal_context_length, window_size):
        size = len(elements)
        total = []
        if size <= temporal_context_length:
            return elements
        for i in range(0, size-temporal_context_length+1, window_size):
            temp = np.array(elements[i:i+temporal_context_length])
            total.append(temp)
        total.append(elements[size-temporal_context_length:size])
        total = np.array(total)
        return total

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item])
        y = torch.tensor(self.y[item])
        return x, y
