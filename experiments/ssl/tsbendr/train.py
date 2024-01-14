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
from experiments.ssl.tsbendr.evaluate import Evaluation
from experiments.ssl.tsbendr.model import TSBendr
from experiments.ssl.tsbendr.model2 import TSBendr_1way
from experiments.ssl.tsbendr.data_loader import batch_dataloader
from dataset.utils import split_train_test_val_files
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as f
import pandas as pd



# warnings.filterwarnings(action='ignore')


random_seed = 424 # my random seed
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
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', 'dataset', 'Sleep-EDFX-2018'), help='data path')

    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])

    # Train (for TSBendr)
    parser.add_argument('--train_epochs', default=300, type=int)
    parser.add_argument('--train_lr_rate', default=0.0005, type=float, help='default was 0.0005')
    parser.add_argument('--train_batch_size', default=256, type=int)

    # TSBendr Hyperparameter
    parser.add_argument('--projection_type', default='linear', type=str, choices=['linear', 'nonlinear'], help='encoder projection type')
    parser.add_argument('--feature_dim', default=256, type=int, help='encoder output dimension')
    parser.add_argument('--projection_hidden', default=512, type=int, help='used for nonlinear projection')
    parser.add_argument('--fft_window', default=64, type=int, choices=[8,16,32,64], help='fft size is 256, contraWR used 64')
    parser.add_argument('--patch_size', default=16, type=int, help='feature_dim should be divided into patch_size') # 16 미만으로 설정하면 메모리에러로 터짐ㅋㅋ
    parser.add_argument('--emb_dim', default=128, type=int, help='transformer embedding dim') # 256 넘어가면 성능저하
    parser.add_argument('--num_layer', default=3, type=int, help='transformer layer num')
    parser.add_argument('--num_head', default=4, type=int, help='emb_dim must be divisible by num_heads') # default is 8
    parser.add_argument('--masking_ratio', default=0.6, type=float, help='0.0 is for evaluation mode')

    # Setting Checkpoint Path~
    parser.add_argument('--print_point', default=5, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', '..', 'ckpt', 'tsbendr'), type=str)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = TSBendr(seq_len=args.sampling_rate * 30, # sampling rate :100 * 30sec -> 3000
                             feature_dim=args.feature_dim,
                             projection_hidden=args.projection_hidden,
                             projection_type=args.projection_type,
                             fft_window=args.fft_window,
                             patch_size=args.patch_size,
                             emb_dim=args.emb_dim,
                             num_layer=args.num_layer,
                             num_head=args.num_head).to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.train_paths, self.ft_paths, self.eval_paths = self.data_paths()
        self.writer_log = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'log.csv')
        self.param_names = [name for name, _ in self.model.named_modules()]


    def train(self):
        print('K-Fold : {}/{}'.format(self.args.n_fold + 1, self.args.k_splits))
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(paths=self.train_paths, batch_size=self.args.train_batch_size)

        # Train (for TSBendr)
        total_loss, total_bf_acc, total_bf_mf1, total_ft_acc, total_ft_mf1 = [], [], [], [], []

        for epoch in range(self.args.train_epochs):
            self.model.train()

            epoch_train_loss = []
            for batch in train_dataloader.gather_async(num_async=5):
                self.optimizer.zero_grad()
                signal = batch.to(device) # raw : (512, 1, 3000) -> (b, c, 30sec*100Hz)
                patch_target, patch_pred, mask = self.model(signal, masking_ratio=self.args.masking_ratio)
                # loss_reconstruct = f.l1_loss(patch_target, patch_pred, reduction='none') # 전체 손실 matrix
                # loss = (loss_reconstruct * mask).sum() / (mask.sum() + 1e-5) # masking 한 부분의 평균손실
                loss = (patch_pred - patch_target) ** 2
                loss = loss.mean(dim=-1)  # [B, N], mean loss per patch
                mask = mask[:,:,0] # [B, N, D] -> D는 어차피 [B, N, 1] 을 D만큼 repeat 한거임 -> 맨 앞에꺼만 추출해도 동일, [B, N] 이됨.

                loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                # print(loss)

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
                epoch_ft_acc, epoch_ft_mf1 = accuracy_score(y_true=epoch_ft_real, y_pred=epoch_ft_pred),\
                                             f1_score(y_true=epoch_ft_real, y_pred=epoch_ft_pred, average='macro')

                print('[Epoch] : {0:03d} \t '
                      '[Train Loss] => {1:.4f} \t '
                      '[Fine Tuning] => Acc: {2:.4f} MF1 {3:.4f} \t'
                      '[Frozen Backbone] => Acc: {4:.4f} MF1: {5:.4f}'.format(
                        epoch + 1, epoch_train_loss, epoch_ft_acc, epoch_ft_mf1, epoch_bf_acc, epoch_bf_mf1))

                self.save_ckpt(epoch, epoch_train_loss, epoch_bf_pred, epoch_bf_real, epoch_ft_pred, epoch_ft_real) # TODO: Save Checkpoint
                total_loss.append(epoch_train_loss)
                total_bf_acc.append(epoch_bf_acc)
                total_bf_mf1.append(epoch_bf_mf1)
                total_ft_acc.append(epoch_ft_acc)
                total_ft_mf1.append(epoch_ft_mf1)
            else:
                print('[Epoch] : {0:03d} \t '
                      '[Train Loss] => {1:.4f}'.format(epoch + 1, epoch_train_loss))

        epoch_labels = [i * self.args.print_point for i in range(len(total_loss))]
        epoch_labels[0] = 1
        df = pd.DataFrame({'epoch': epoch_labels,
                           'loss': total_loss,
                           'bf_acc': total_bf_acc, 'bf_mf1': total_bf_mf1,
                           'ft_acc': total_ft_acc, 'ft_mf1': total_ft_mf1})
        df.to_csv(self.writer_log, index=False)
        ray.shutdown()

    def save_ckpt(self, epoch, train_loss, bf_pred, bf_real, ft_pred, ft_real):
        if not os.path.exists(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model')):
            os.makedirs(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model'))

        ckpt_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model', '{0:04d}.pth'.format(epoch+1))
        torch.save({
            'epoch': epoch+1,
            'backbone_name': 'tsbendr',
            'backbone_parameter': {
                'input_size': self.args.sampling_rate * 30, 'patch_size': self.args.patch_size,
                'emb_dim': self.args.emb_dim, 'num_layer': self.args.num_layer,
                'num_head': self.args.num_head, 'mask_ratio': self.args.mask_ratio,
                'feature_dim': self.args.feature_dim, 'projection_hidden': self.args.projection_hidden,
                'projection_type': self.args.projection_type
            },
            'model_state': self.model.state_dict(),
            'hyperparameter': self.args.__dict__,
            'loss': train_loss,
            'result': {
                'frozen_backbone': {'real': bf_real, 'pred': bf_pred},
                'fine_tuning': {'real': ft_real, 'pred': ft_pred}
            },
            'paths': {'train_paths': self.train_paths, 'ft_paths': self.ft_paths, 'eval_paths': self.eval_paths}
        }, ckpt_path)

    def compute_evaluation(self):
        # 1. Backbone Frozen (freeze encoder, train decoder + fc)
        evaluation = Evaluation(backbone=self.model, device=device)
        bf_pred, bf_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=self.param_names)
        del evaluation

        # 2. Fine Tuning (freeze encoder + decoder, train fc)
        evaluation = Evaluation(backbone=self.model, device=device)
        ft_pred, ft_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=self.param_names[:((-21) * 2) - 1]) # Transformer Block 한개당 21개 layer 가짐. -> 2개 Block freeze, 맨 마지막 layer인 layer norm 포함
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

