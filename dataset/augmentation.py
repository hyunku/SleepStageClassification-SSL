# -*- coding:utf-8 -*-
import copy
import random
import numpy as np
from tslearn.preprocessing import TimeSeriesResampler
from dataset.utils import butter_bandpass_filter


random_seed = 777
np.random.seed(random_seed)
random.seed(random_seed)


class SignalAugmentation(object):
    # https://arxiv.org/pdf/2109.07839.pdf
    # https://arxiv.org/pdf/1706.00527.pdf

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.second = 30
        self.input_length = self.second * self.sampling_rate
        self.l_freq, self.h_freq = 1, 50

        self.band_size = 15                                     # for random_bandpass_filter
        self.gn_scaling = list(np.arange(0.05, 0.15, 0.01))     # for %5 ~ 15% gaussian noise
        self.permutation_min, self.permutation_max = 3, 6       # for permutation
        self.window_width = int(self.sampling_rate / 2)         # window width
        self.cutout_size = int(self.input_length / 2)

    def process(self, x, aug_name, p=0.5):
        x = copy.deepcopy(x)
        if aug_name == 'none':
            return x
        if aug_name == 'random_crop':
            x = self.random_crop(x, p)
            return x
        elif aug_name == 'random_bandpass_filter':
            x = self.random_bandpass_filter(x, p)
            return x
        elif aug_name == 'random_gaussian_noise':
            x = self.random_gaussian_noise(x, p)
            return x
        elif aug_name == 'random_horizontal_flip':
            x = self.random_horizontal_flip(x, p)
            return x
        elif aug_name == 'random_permutation':
            x = self.random_permutation(x, p)
            return x
        elif aug_name == 'random_temporal_cutout':
            x = self.random_temporal_cutout(x, p)
            return x
        else:
            raise NotImplementedError()

    def random_crop(self, x, p=0.5):
        sr = int(self.input_length / 4)
        if random.random() < p:
            index_1 = np.random.randint(low=0, high=sr - 1, size=1)[0]
            index_2 = np.random.randint(low=self.input_length - sr, high=self.input_length - 1, size=1)[0]
            x = x[:, index_1:index_2]  # 1. Crop
            x = TimeSeriesResampler(sz=self.input_length).fit_transform(x)  # 2. Resample
            x = np.squeeze(x, axis=-1)
        return x

    def random_bandpass_filter(self, x, p=0.5):
        low_cut_range = list(range(self.l_freq, self.l_freq + self.band_size))
        high_cut_range = list(range(self.h_freq - self.band_size, self.h_freq))

        if random.random() < p:
            low_cut, high_cut = np.random.choice(low_cut_range, 1)[0], np.random.choice(high_cut_range, 1)[0]
            x = butter_bandpass_filter(x, fs=self.sampling_rate, low_cut=low_cut, high_cut=high_cut)
        return x

    def random_gaussian_noise(self, x, p=0.5):
        mu = 0.0
        x = np.array(x)
        std = np.random.choice(self.gn_scaling, 1)[0] * np.std(x)

        if random.random() < p:
            noise = np.random.normal(mu, std, x.shape)
            x = x + noise
        return x

    @staticmethod
    def random_horizontal_flip(x, p=0.5):
        if random.random() < p:
            x = np.flip(x, axis=-1)
        return x

    def random_permutation(self, x, p=0.5):
        if random.random() < p:
            num_segment = np.random.randint(self.permutation_min, self.permutation_max + 1, size=1).reshape(-1)[0]
            indexes = np.random.choice(self.input_length, num_segment - 1, replace=False)
            indexes = list(np.sort(indexes))
            indexes = [0] + indexes + [self.input_length]
            samples = []
            for index_1, index_2 in zip(indexes[:-1], indexes[1:]):
                samples.append(x[:, index_1:index_2])

            nx = []
            for index in np.random.permutation(np.arange(num_segment)):
                nx.append(samples[index])
            x = np.concatenate(nx, axis=-1)
        return x

    def random_temporal_cutout(self, x, p):
        start_list = list(np.arange(0, self.input_length))
        start = np.random.choice(start_list, 1)[0]

        width_list = list(np.arange(start, start+self.cutout_size))
        width = np.random.choice(width_list, 1)[0]
        end = (self.input_length if width > self.input_length else width)
        if random.random() < p:
            std = np.mean(x)
            x[:, start:end] = std
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d1 = np.random.randn(1, 3000)
    # plt.plot(d1.reshape(-1))
    augment = SignalAugmentation(sampling_rate=100)
    d2 = augment.process(x=d1, aug_name='random_temporal_cutout', p=0.99)
    plt.plot(d1.reshape(-1))
    plt.plot(d2.reshape(-1))
    plt.show()