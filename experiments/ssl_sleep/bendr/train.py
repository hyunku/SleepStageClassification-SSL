# -*- coding:utf-8 -*-
import os
import sys

import mne
import argparse
import pandas as pd
import torch.optim as opt

from experiments.ssl_sleep.bendr.model import BENDR
from models.utils import get_backbone_parameter
from experiments.ssl_sleep.bendr.evaluate import Evaluation
from experiments.ssl_sleep.bendr.data_loader import *
from experiments.ssl_sleep.bendr.data_loader import batch_dataloader
from dataset.utils import split_train_test_val_files
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
from torchsummary import summary
from torch.optim.lr_scheduler import LambdaLR
import math

warnings.filterwarnings(action='ignore')

random_seed = 424
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--sampling_rate', default=100) # input data (sequence) : 3000 -> 100Hz * 30sec
    parser.add_argument('--seq_size', default=15000, help='for dataset concatenate') # 1 means 30 sec, 2means 60 sec
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', 'dataset', 'Sleep-EDFX-2018'), help='data path')
    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])

    # Train (for BENDR)
    parser.add_argument('--train_epochs', default=300, type=int) # default : 300
    parser.add_argument('--train_lr_rate', default=0.00005, type=float)
    parser.add_argument('--warmup_rate', default=0.03, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int) # 30sec : 256, 60 sec : 128, 150sec : 32
    parser.add_argument('--temperature', default=0.1, type=float) # don't fix it
    parser.add_argument('--mask_rate', default=0.1, type=float) # don't fix it
    parser.add_argument('--mask_span', default=6, type=int, help='masking window size') # default : 6
    parser.add_argument('--num_negatives', default=20, type=int) # default : 100
    parser.add_argument('--backbone_name', default='BENDREncoder', type=str)
    parser.add_argument('--backbone_parameter', default=get_backbone_parameter(model_name='BENDREncoder',
                                                                               sampling_rate=100))

    # Contextualizer parameters
    parser.add_argument('--context_dim', default=3076, type=int, help='transformer dimension') # 3076
    parser.add_argument('--context_heads', default=8, type=int, help='transformer heads') # 8
    parser.add_argument('--context_layers', default=8, type=int, help='transformer layers') # 8
    parser.add_argument('--context_dropouts', default=0.15, type=float, help='transformer dropouts') # 0.15


    # Setting Checkpoint Path
    parser.add_argument('--print_point', default=50, type=int) # TODO: default is 20
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt',
                                                            'SHHS', 'BENDR', 'CNNEncoder2D_SLEEP'), type=str)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = BENDR(backbone_name=self.args.backbone_name,
                           backbone_parameter=self.args.backbone_parameter,
                           mask_span=self.args.mask_span,
                           mask_rate=self.args.mask_rate,
                           num_negatives=self.args.num_negatives,
                           temperature=self.args.temperature,
                           context_dim=self.args.context_dim,
                           context_heads=self.args.context_heads,
                           context_layers=self.args.context_layers,
                           context_dropouts=self.args.context_dropouts).to(device)
        self.warmup_epochs = self.args.warmup_rate * self.args.train_epochs
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.lr_lambda = lambda epoch: warmup_cosine_schedule(epoch, self.warmup_epochs, self.args.train_epochs)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.train_paths, self.ft_paths, self.eval_paths = self.data_paths()
        self.writer_log = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'log.csv')
        self.params = [name for name, _ in self.model.named_parameters()]  # whole params
        self.encoder_params = [name for name, _ in self.model.encoder.named_parameters()]  # encoder params for evaluation
        # print(self.encoder_params) # print and check for freeze layers


    def train(self):
        print('K-Fold : {}/{}'.format(self.args.n_fold + 1, self.args.k_splits))
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(paths=self.train_paths, batch_size=self.args.train_batch_size, seq_size=self.args.seq_size)
        # Train (for BENDR)
        total_loss, total_bf_acc, total_bf_mf1, total_ft_acc, total_ft_mf1 = [], [], [], [], []
        for epoch in range(self.args.train_epochs):
            self.model.train()

            epoch_train_loss = []
            for batch in train_dataloader.gather_async(num_async=5):
                self.optimizer.zero_grad()
                x = batch.to(device) # (b, 1, 1, 3000)
                logit, label, mask = self.model(x)  # 1024, 256 -> b, f_dim
                loss = self.criterion(logit, label)
                epoch_train_loss.append(float(loss.detach().cpu().item()))

                loss.backward()
                self.optimizer.step()

            epoch_train_loss = np.mean(epoch_train_loss)
            self.scheduler.step()

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

                # self.save_ckpt(epoch, epoch_train_loss, epoch_bf_pred, epoch_bf_real, epoch_ft_pred, epoch_ft_real) # TODO: For test
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
        # df.to_csv(self.writer_log, index=False) # TODO: For test
        ray.shutdown()

    def save_ckpt(self, epoch, train_loss, bf_pred, bf_real, ft_pred, ft_real):
        if not os.path.exists(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model')):
            os.makedirs(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model'))

        ckpt_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model', '{0:04d}.pth'.format(epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'backbone_name': self.args.backbone_name,
            'backbone_parameter': self.args.backbone_parameter,
            'model_state': self.model.encoder.encoder.state_dict(),
            'hyperparameter': self.args.__dict__,
            'loss': train_loss,
            'result': {
                'frozen_backbone': {'real': bf_real, 'pred': bf_pred},
                'fine_tuning': {'real': ft_real, 'pred': ft_pred}
            },
            'paths': {'train_paths': self.train_paths, 'ft_paths': self.ft_paths, 'eval_paths': self.eval_paths}
        }, ckpt_path)

    def compute_evaluation(self):
        # 1. Backbone Frozen (train only fc) -> Linear Evaluation
        print("Strat evaluation")
        evaluation = Evaluation(backbone=self.model.encoder, device=device)
        bf_pred, bf_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=self.encoder_params)
        del evaluation

        # 2. Fine Tuning (train last layer + fc)
        evaluation = Evaluation(backbone=self.model.encoder, device=device)
        ft_pred, ft_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=self.encoder_params[:13])
        del evaluation
        return (bf_pred, bf_real), (ft_pred, ft_real)

    def data_paths(self):
        kf = split_train_test_val_files(base_path=self.args.base_path, n_splits=self.args.k_splits)
        paths = kf[self.args.n_fold]
        train_paths, ft_paths, eval_paths = paths['train_paths'], paths['ft_paths'], paths['eval_paths']
        train_paths, ft_paths, eval_paths = train_paths[:1], ft_paths[:1], eval_paths[:1]
        return train_paths, ft_paths, eval_paths

def warmup_cosine_schedule(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        max_decay_epochs = total_epochs - warmup_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / max_decay_epochs))
        return cosine_decay


if __name__ == '__main__':
    augments = get_args()
    for n_fold in range(augments.k_splits):
        augments.n_fold = n_fold
        trainer = Trainer(augments)
        trainer.train()
