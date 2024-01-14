# -*- coding:utf-8 -*-
import os
import sys

import ray
import torch
import random
import mne
import numpy as np
import argparse
import torch.optim as opt
from experiments.ssl.moco.evaluate import Evaluation
# from experiments.ssl.moco.model import MoCo # ver.2
from experiments.ssl.moco.model2 import MoCo # ver.3
from experiments.ssl.moco.data_loader import batch_dataloader
from dataset.utils import split_train_test_val_files
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as f
import pandas as pd
import matplotlib.pyplot as plt
from models.utils import get_backbone_parameter

# warnings.filterwarnings(action='ignore')


random_seed = 424  # my random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')  # to_device로 변수, 모델 두개만 보내면 됌(필수). 그 외 loss function 등등 연산이 필요한 것은 보내도 되고 안보내도됌


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--sampling_rate', default=100)
    # parser.add_argument('--base_path', default=os.path.join('D:/hyunku', 'Sleep-EDFX-2018'), help='data path')
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', 'dataset', 'Sleep-EDFX-2018'), help='data path')

    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--labels', default=['W', 'N1', 'N2', 'N3', 'R'])

    # Train (for TSBendr)
    parser.add_argument('--train_epochs', default=300, type=int)
    parser.add_argument('--train_lr_rate', default=0.005, type=float, help='default was 0.0005')
    parser.add_argument('--train_batch_size', default=256, type=int)  # 1024 초과로 설정하면 메모리 에러로 터짐ㅋㅋ

    # Hyperparameters
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--queue_dim', default=65536, type=int) # default is 65536
    parser.add_argument('--backbone_name', default='CNNEncoder2D_SLEEP', type=str, choices=['EEGNet', 'ShallowConvNet', 'BaseCNN'])
    parser.add_argument('--backbone_parameter', default=get_backbone_parameter(model_name='CNNEncoder2D_SLEEP',
                                                                               sampling_rate=100))
    parser.add_argument('--augmentations', default=[('random_permutation', 0.95),
                                                    ('random_bandpass_filter', 0.95)])
    parser.add_argument('--projection_hidden', default=512, type=int)
    parser.add_argument('--projection_size', default=128, type=int)

    # Setting Checkpoint Path~
    parser.add_argument('--print_point', default=20, type=int)  # default is 20
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', '..', 'ckpt',
                                                            'Sleep-EDFX', 'tsbendr'), type=str)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = MoCo(backbone_name=self.args.backbone_name,
                          backbone_parameter=self.args.backbone_parameter,
                          projection_hidden=self.args.projection_hidden,
                          projection_size=self.args.projection_size,
                          temperature=self.args.temperature,
                          queue_dim=self.args.queue_dim).to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.train_paths, self.ft_paths, self.eval_paths = self.data_paths()
        self.writer_log = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'log.csv')
        # self.param_names = [name for name, _ in self.model.named_parameters()]
        self.encoder_param_names = [name for name, _ in self.model.encoder_q.named_parameters()]


    def train(self):
        print('K-Fold : {}/{}'.format(self.args.n_fold + 1, self.args.k_splits))
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(paths=self.train_paths, batch_size=self.args.train_batch_size,
                                            augmentations=self.args.augmentations, freqs=self.args.sampling_rate)

        # Train (for TSBendr)
        total_loss, total_bf_acc, total_bf_mf1, total_ft_acc, total_ft_mf1 = [], [], [], [], []

        for epoch in range(self.args.train_epochs):
            self.model.train()

            epoch_train_loss = []
            for batch in train_dataloader.gather_async(num_async=5):
                self.optimizer.zero_grad()
                x1, x2 = batch
                x1, x2 = x1.to(device), x2.to(device)
                # if x1.shape[0] < self.args.train_batch_size: # 큐 사이즈에 맞추기 위함. used for V2.
                #     continue

                loss = self.model((x1, x2), mode='train')
                epoch_train_loss.append(float(loss.detach().cpu().item()))

                # self.optimizer.zero_grad() # 맨 위에다 해줬음
                loss.backward()
                self.optimizer.step()

            epoch_train_loss = np.mean(epoch_train_loss)

            # if (epoch + 1) % self.args.print_point == 0 or epoch == 0:
            if (epoch + 1) % self.args.print_point == 0:
                # Print Log & Save Checkpoint Path
                (epoch_bf_pred, epoch_bf_real), (epoch_ft_pred, epoch_ft_real) = self.compute_evaluation()

                # Calculation Metric (Acc & MF1)
                epoch_bf_acc, epoch_bf_mf1 = accuracy_score(y_true=epoch_bf_real, y_pred=epoch_bf_pred), \
                                             f1_score(y_true=epoch_bf_real, y_pred=epoch_bf_pred, average='macro')
                epoch_ft_acc, epoch_ft_mf1 = accuracy_score(y_true=epoch_ft_real, y_pred=epoch_ft_pred), \
                                             f1_score(y_true=epoch_ft_real, y_pred=epoch_ft_pred, average='macro')

                print('[Epoch] : {0:03d} \t '
                      '[Train Loss] => {1:.4f} \t '
                      '[Fine Tuning] => Acc: {2:.4f} MF1 {3:.4f} \t'
                      '[Frozen Backbone] => Acc: {4:.4f} MF1: {5:.4f}'.format(
                    epoch + 1, epoch_train_loss, epoch_ft_acc, epoch_ft_mf1, epoch_bf_acc, epoch_bf_mf1))

                # self.save_ckpt(epoch, epoch_train_loss, epoch_bf_pred, epoch_bf_real, epoch_ft_pred, epoch_ft_real) # TODO: Save Checkpoint
                total_loss.append(epoch_train_loss)
                total_bf_acc.append(epoch_bf_acc)
                total_bf_mf1.append(epoch_bf_mf1)
                total_ft_acc.append(epoch_ft_acc)
                total_ft_mf1.append(epoch_ft_mf1)
            else:
                print('[Epoch] : {0:03d} \t '
                      '[Train Loss] => {1:.4f}'.format(epoch + 1, epoch_train_loss))

    #     epoch_labels = [i * self.args.print_point for i in range(len(total_loss))]
    #     epoch_labels[0] = 1
    #     df = pd.DataFrame({'epoch': epoch_labels,
    #                        'loss': total_loss,
    #                        'bf_acc': total_bf_acc, 'bf_mf1': total_bf_mf1,
    #                        'ft_acc': total_ft_acc, 'ft_mf1': total_ft_mf1})
    #     df.to_csv(self.writer_log, index=False)
    #     ray.shutdown()
    #
    # def save_ckpt(self, epoch, train_loss, bf_pred, bf_real, ft_pred, ft_real):
    #     if not os.path.exists(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model')):
    #         os.makedirs(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model'))
    #
    #     ckpt_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model', '{0:04d}.pth'.format(epoch + 1))
    #     torch.save({
    #         'epoch': epoch + 1,
    #         'backbone_name': 'tsbendr',
    #         'backbone_parameter': {
    #             'input_size': self.args.sampling_rate * 30, 'patch_size': self.args.patch_size,
    #             'emb_dim': self.args.emb_dim, 'num_layer': self.args.num_layer,
    #             'num_head': self.args.num_head, 'mask_ratio': self.args.mask_ratio,
    #             'feature_dim': self.args.feature_dim, 'projection_hidden': self.args.projection_hidden,
    #             'projection_type': self.args.projection_type
    #         },
    #         'model_state': self.model.state_dict(),
    #         'hyperparameter': self.args.__dict__,
    #         'loss': train_loss,
    #         'result': {
    #             'frozen_backbone': {'real': bf_real, 'pred': bf_pred},
    #             'fine_tuning': {'real': ft_real, 'pred': ft_pred}
    #         },
    #         'paths': {'train_paths': self.train_paths, 'ft_paths': self.ft_paths, 'eval_paths': self.eval_paths}
    #     }, ckpt_path)

    def compute_evaluation(self):
        # 1. Backbone Frozen (freeze encoder, only train fc)
        evaluation = Evaluation(backbone=self.model.encoder_q, device=device)
        bf_pred, bf_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=self.encoder_param_names)
        del evaluation

        # 2. Fine Tuning (freeze few encoder layer + train fc)
        evaluation = Evaluation(backbone=self.model.encoder_q, device=device)
        ft_pred, ft_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=self.encoder_param_names[:16])
        del evaluation
        return (bf_pred, bf_real), (ft_pred, ft_real)

    def data_paths(self):
        kf = split_train_test_val_files(base_path=self.args.base_path, n_splits=self.args.k_splits)
        paths = kf[self.args.n_fold]
        train_paths, ft_paths, eval_paths = paths['train_paths'], paths['ft_paths'], paths['eval_paths']
        # return train_paths[:20], ft_paths[:20], eval_paths[:20] # 현재 test용으로 5개 데이터셋만 인덱싱해서 사용
        return train_paths, ft_paths, eval_paths


if __name__ == '__main__':
    augments = get_args()
    print(f"CUDA Availablity: {torch.cuda.is_available()}")
    for n_fold in range(augments.k_splits):
        augments.n_fold = n_fold
        trainer = Trainer(augments)
        trainer.train()

