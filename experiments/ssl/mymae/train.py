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
from experiments.ssl.mymae.evaluate import Evaluation
from experiments.ssl.mymae.model import MAE
from experiments.ssl.mymae.data_loader import batch_dataloader
from dataset.utils import split_train_test_val_files
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as f
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR, StepLR
import math
from torch.optim.lr_scheduler import LambdaLR

# warnings.filterwarnings(action='ignore')


random_seed = 424  # my random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO: 너무 높은 학습률은 gradient exploding, 너무 낮은 학습률은 gradient vanishing 초래 가능 => 모델의 gradient 최소/최대값 찍어보기

# device = torch.device('cpu')  # to_device로 변수, 모델 두개만 보내면 됌(필수). 그 외 loss function 등등 연산이 필요한 것은 보내도 되고 안보내도됌


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--sampling_rate', default=100)
    # parser.add_argument('--base_path', default=os.path.join('D:/hyunku', 'Sleep-EDFX-2018'), help='data path')
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', 'dataset', 'Sleep-EDFX-2018'), help='data path')

    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])

    # Train (for MAE)
    parser.add_argument('--train_epochs', default=1000, type=int)
    parser.add_argument('--train_lr_rate', default=0.0005, type=float, help='default was 0.0005')
    parser.add_argument('--warmup_rate', default=0.3, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)


    # MAE Hyperparameter
    parser.add_argument('--patch_size', default=30, type=int, help='input_size(sampling_rate * 30) should be divided into patch_size')
    parser.add_argument('--emb_dim', default=512, type=int, help='vit encoder embedding dim') # mae: 1024
    parser.add_argument('--num_layer', default=8, type=int, help='vit encoder layer num') # mae: 24
    parser.add_argument('--num_head', default=8, type=int, help='emb_dim must be divisible by num_heads, for encoder')  # mae: 16
    parser.add_argument('--mlp_ratio', default=4, type=int, help='vit fc hidden layer ratio -> hidden layer : emb_dim * 0.4') # mae: .4
    parser.add_argument('--dec_embed_dim', default=256, type=int, help='vit decoder embedding dim') # mae: 512
    parser.add_argument('--dec_num_layer', default=6, type=int, help='vit decoder layer num') # mae: 8
    parser.add_argument('--dec_num_head', default=8, type=int, help='for decoder, must be divisible by dec_emb_dim') # mae: 16
    parser.add_argument('--masking_ratio', default=0.75, type=float, help='0.0 is for evaluation mode') # mae: 0.75

    # Setting Checkpoint Path~
    parser.add_argument('--print_point', default=20, type=int)  # default is 20
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', '..', 'ckpt',
                                                            'Sleep-EDFX', 'tsbendr'), type=str)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = MAE(input_size=args.sampling_rate * 30,  # sampling rate :100 * 30sec -> 3000
                         patch_size=args.patch_size,
                         embed_dim=args.emb_dim,
                         num_layer=args.num_layer,
                         num_heads=args.num_head,
                         mlp_ratio=args.mlp_ratio,
                         dec_embed_dim=args.dec_embed_dim,
                         dec_num_layer=args.dec_num_layer,
                         dec_num_heads=args.dec_num_head).to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.warmup_epochs = self.args.warmup_rate * self.args.train_epochs
        self.lr_lambda = lambda epoch: warmup_cosine_schedule(epoch, self.warmup_epochs, self.args.train_epochs)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        self.criterion = torch.nn.MSELoss()
        self.train_paths, self.ft_paths, self.eval_paths = self.data_paths()
        self.writer_log = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'log.csv')
        self.param_names = [name for name, _ in self.model.named_parameters()] # whole params
        # self.encoder_params = [name for name, _ in self.model.named_modules() if name.split('.')[0] == 'encoder']
        self.encoder_params = [name for name, _ in self.model.encoder.named_parameters()] # encoder params for evaluation



    def train(self):
        print('K-Fold : {}/{}'.format(self.args.n_fold + 1, self.args.k_splits))
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(paths=self.train_paths, batch_size=self.args.train_batch_size)

        # Train (for TSBendr)
        total_loss, total_bf_acc, total_bf_mf1, total_ft_acc, total_ft_mf1 = [], [], [], [], []

        for epoch in range(self.args.train_epochs):
            self.model.train()

            epoch_train_loss = []
            for batch_idx, batch in enumerate(train_dataloader.gather_async(num_async=5), start=0):
                self.optimizer.zero_grad()
                signal = batch.to(device)  # raw : (512, 1, 1, 3000) -> (b, c, h, w(30sec*100Hz)) = (b, window(sequence), channels(features), samples)
                # 1은 masking 진행!! 0은 masking 진행 X!!

                origin_sig, pred_sig, mask = self.model(signal, mask_ratio=self.args.masking_ratio) # 셋 다 (b 3000)
                loss = self.criterion(pred_sig * mask, origin_sig * mask) / self.args.masking_ratio

                epoch_train_loss.append(float(loss.detach().cpu().item()))
                loss.backward()
                self.optimizer.step()

                # Plotting
                batch_show = 0 # epoch 중 1번째 batch 데이터를 보겠다.
                if (epoch + 1) % self.args.print_point == 0 and batch_idx == batch_show: # epoch 중 1번째 batch의 데이터만 시각화
                    plt.figure(figsize=(15, 10)) # plt 창 생성

                    # 예측된 신호(pred_sig) 그래프
                    plt.plot(pred_sig[batch_show].cpu().detach().numpy(), label="Predicted Signal", alpha=0.7)

                    # 원본 신호(signal) 그래프
                    plt.plot(origin_sig[batch_show].cpu().detach().numpy(), label="Original Signal", linewidth=2.0, alpha=0.7)

                    # mask가 1인 부분만 배경을 어둡게 칠함 -> mask 0은 masking X, 1이 masking O
                    for i in range(mask.shape[1]):
                        if mask[batch_show, i] == 1:
                            plt.axvspan(i, i + 1, facecolor='gray', alpha=0.5)

                    plt.legend()
                    plt.title(f"Epoch {epoch + 1}")
                    plt.show()

            epoch_train_loss = np.mean(epoch_train_loss)
            # self.scheduler.step()

            # if (epoch + 1) % self.args.print_point == 0 or epoch == 0:
            if (epoch + 1) % self.args.print_point == 0:
                print("start evaluation")

                # Print Log & Save Checkpoint Path
                (epoch_bf_pred, epoch_bf_real), (epoch_ft_pred, epoch_ft_real) = self.compute_evaluation()

                # Calculation Metric (Acc & MF1)
                epoch_bf_acc, epoch_bf_mf1 = accuracy_score(y_true=epoch_bf_real, y_pred=epoch_bf_pred), \
                                             f1_score(y_true=epoch_bf_real, y_pred=epoch_bf_pred, average='macro')
                epoch_ft_acc, epoch_ft_mf1 = accuracy_score(y_true=epoch_ft_real, y_pred=epoch_ft_pred), \
                                             f1_score(y_true=epoch_ft_real, y_pred=epoch_ft_pred, average='macro')

                print('[Epoch] : {0:03d} \t '
                      '[Train Loss] => {1:.8f} \t '
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
                      '[Train Loss] => {1:.8f}'.format(epoch + 1, epoch_train_loss))

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

        ckpt_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model', '{0:04d}.pth'.format(epoch + 1))
        torch.save({
            'epoch': epoch + 1,
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
        # 1. Backbone Frozen => Linear Evaluation
        evaluation = Evaluation(backbone=self.model.encoder, device=device)
        bf_pred, bf_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=self.encoder_params) # linear evaluation
        del evaluation

        # 2. Fine Tuning (freeze encoder + decoder, train fc)
        evaluation = Evaluation(backbone=self.model.encoder, device=device)
        ft_pred, ft_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=self.encoder_params[:((-12) * 2)])  # Transformer Block 한개당 12개 layer 가짐. -> 2개 Block freeze
        del evaluation
        return (bf_pred, bf_real), (ft_pred, ft_real)

    def data_paths(self):
        kf = split_train_test_val_files(base_path=self.args.base_path, n_splits=self.args.k_splits)
        paths = kf[self.args.n_fold]
        train_paths, ft_paths, eval_paths = paths['train_paths'], paths['ft_paths'], paths['eval_paths']
        return train_paths[:1], ft_paths[:1], eval_paths[:1]


def warmup_cosine_schedule(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        max_decay_epochs = total_epochs - warmup_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / max_decay_epochs))
        return cosine_decay


if __name__ == '__main__':
    augments = get_args()
    print(f"CUDA Availablity: {torch.cuda.is_available()}")
    for n_fold in range(augments.k_splits):
        augments.n_fold = n_fold
        trainer = Trainer(augments)
        trainer.train()


