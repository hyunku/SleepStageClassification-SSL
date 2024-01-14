# -*- coding:utf-8 -*-
import os
# from dataset.data_parser import *

# train_items = [
#     os.path.join('/home/brainlab/Dataset2/DeepBCI/MI/block', 'sess01_subj{0:02d}_EEG_MI.npz'.format(i+1))
#     for i in range(54)
# ]
#
# ft_items = [
#     os.path.join('/home/brainlab/Dataset2/DeepBCI/MI/block', 'sess02_subj{0:02d}_EEG_MI.npz'.format(i+1))
#     for i in range(5, 54)
# ]
#
# eval_items = [
#     os.path.join('/home/brainlab/Dataset2/DeepBCI/MI/block', 'sess02_subj{0:02d}_EEG_MI.npz'.format(i+1))
#     for i in range(5)
# ]

train_items = [
    os.path.join('/home/brainlab/Workspace/Chlee/SSL_BCI_EEG/data/SSVEP/OpenBMI',
                 'sess01_subj{0:02d}_EEG_SSVEP.pkl'.format(i + 1))
    for i in range(54)
]

ft_items = [
    os.path.join('/home/brainlab/Workspace/Chlee/SSL_BCI_EEG/data/SSVEP/OpenBMI',
                 'sess02_subj{0:02d}_EEG_SSVEP.pkl'.format(i + 1))
    for i in range(5, 54)
]

eval_items = [
    os.path.join('/home/brainlab/Workspace/Chlee/SSL_BCI_EEG/data/SSVEP/OpenBMI',
                 'sess02_subj{0:02d}_EEG_SSVEP.pkl'.format(i + 1))
    for i in range(5)
]

# train_items = [
#     os.path.join('/home/brainlab/Workspace/Chlee/SSL_BCI_EEG/data/MI/OpenBMI',
#                  'sess01_subj{0:02d}_EEG_MI.pkl'.format(i + 1))
#     for i in range(54)
# ]
#
# ft_items = [
#     os.path.join('/home/brainlab/Workspace/Chlee/SSL_BCI_EEG/data/MI/OpenBMI',
#                  'sess02_subj{0:02d}_EEG_MI.pkl'.format(i + 1))
#     for i in range(5, 54)
# ]
#
# eval_items = [
#     os.path.join('/home/brainlab/Workspace/Chlee/SSL_BCI_EEG/data/MI/OpenBMI',
#                  'sess02_subj{0:02d}_EEG_MI.pkl'.format(i + 1))
#     for i in range(5)
# ]

# train_items = [
#     os.path.join('/home/brainlab/Workspace/Chlee/SSL_BCI_EEG/data/MI/GIST',
#                  's{0:02d}.pkl'.format(i + 1))
#     for i in range(40)
# ]
#
# ft_items = [
#     os.path.join('/home/brainlab/Workspace/Chlee/SSL_BCI_EEG/data/MI/GIST',
#                  's{0:02d}.pkl'.format(i + 1))
#     for i in range(40, 45)
# ]
#
# eval_items = [
#     os.path.join('/home/brainlab/Workspace/Chlee/SSL_BCI_EEG/data/MI/GIST',
#                  's{0:02d}.pkl'.format(i + 1))
#     for i in range(45, 52)
# ]


seconds = 4
sampling_rate = 500
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
            'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
            'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4',
            'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h',
            'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h',
            'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']

labels = {'SSVEP/5.45Hz': 0, 'SSVEP/6.67Hz': 1, 'SSVEP/8.57Hz': 2, 'SSVEP/12Hz': 3}
# labels = {'MI/Left-Hand': 0, 'MI/Right-Hand': 1}
