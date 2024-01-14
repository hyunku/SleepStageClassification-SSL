# -*- coding:utf-8 -*-
import os
import torch
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=os.path.join('..', 'ckpt', 'Sleep-EDFX', 'simclr', 'BaseCNN'))
    return parser.parse_args()


def name_change(args):
    pass


def result1(args):
    # Calculation for Sleep Stage Classification Performance
    model_path = args.model_path

    temp = {'bf_acc': [], 'bf_mf1': [], 'bf_kappa': [], 'ft_acc': [], 'ft_mf1': [], 'ft_kappa': []}
    for path in os.listdir(model_path):
        path_ = os.path.join(model_path, path)
        try:
            log_df = pd.read_csv(os.path.join(path_, 'log.csv'))
        except FileNotFoundError:
            continue

        epoch = log_df[log_df['ft_acc'] == log_df['ft_acc'].max()]['epoch'].values[0]
        model_path_ = os.path.join(model_path, path, 'model', '{0:04d}.pth'.format(epoch))

        ckpt = torch.load(model_path_)['result']
        bf_real, ft_real = ckpt['frozen_backbone']['real'], ckpt['fine_tuning']['real']
        bf_pred, ft_pred = ckpt['frozen_backbone']['pred'], ckpt['fine_tuning']['pred']

        bf_acc, bf_mf1, bf_kappa = accuracy_score(y_pred=bf_pred, y_true=bf_real), \
                                   f1_score(y_pred=bf_pred, y_true=bf_real, average='macro'), \
                                   cohen_kappa_score(bf_pred, bf_real)
        ft_acc, ft_mf1, ft_kappa = accuracy_score(y_pred=ft_pred, y_true=ft_real), \
                                   f1_score(y_pred=ft_pred, y_true=ft_real, average='macro'), \
                                   cohen_kappa_score(ft_pred, ft_real)
        temp['bf_acc'].append(bf_acc)
        temp['bf_mf1'].append(bf_mf1)
        temp['bf_kappa'].append(bf_kappa)
        temp['ft_acc'].append(ft_acc)
        temp['ft_mf1'].append(ft_mf1)
        temp['ft_kappa'].append(ft_kappa)
    temp_df = pd.DataFrame(temp)
    print(temp_df.mean() * 100)
    print(temp_df.std() * 100)


if __name__ == '__main__':
    augment = get_args()
    result1(augment)


