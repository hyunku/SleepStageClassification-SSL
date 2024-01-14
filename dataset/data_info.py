# -*- coding:utf-8 -*-


OpenBMI_MI_INFO = {
    'paradigm': 'MI',
    'ch_names': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
                 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4',
                 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h',
                 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h',
                 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'],
    'sfreq': 1000,
    'labels': {'left': 'MI/Left-Hand', 'right': 'MI/Right-Hand'},
    'second': 4
}

OpenBMI_SSVEP_INFO = {
    'paradigm': 'SSVEP',
    'ch_names': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
                 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4',
                 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h',
                 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h',
                 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'],
    'sfreq': 1000,
    'labels': {'down': 'SSVEP/5.45Hz', 'right': 'SSVEP/6.67Hz', 'left': 'SSVEP/8.57Hz', 'up': 'SSVEP/12Hz'},
    'second': 4
}


OpenBMI_ERP_INFO = {
    'paradigm': 'ERP',
    'ch_names': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
                 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4',
                 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h',
                 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h',
                 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'],
    'sfreq': 1000,
    'labels': {'target': 'ERP/Target', 'nontarget': 'ERP/NonTarget'},
    'second': 4
}


GIST_MI_INFO = {
    'paradigm': 'MI',
    'ch_names': ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
                 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'FPz', 'FP2',
                 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
                 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
    'sfreq': 512,
    'labels': {0: 'MI/Left-Hand', 1: 'MI/Right-Hand'},
    'second': 3
}


BCICompetition2A_INFO = {
    'paradigm': 'MI',
    'ch_names': ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                 'P1', 'Pz', 'P2', 'POz'],
    'labels': {1: 'MI/Left-Hand', 2: 'MI/Right-Hand', 3: 'MI/Foot', 4: 'MI/Tongue'},
    'sfreq': 250,
    'second': 3
}
