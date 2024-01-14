# -*- coding:utf-8 -*-
import os
import mne
import argparse
import pandas as pd
import torch.optim as opt

from experiments.supervised.u_sleep.model import USleep
from experiments.supervised.u_sleep.data_loader import *
from experiments.supervised.u_sleep.data_loader import batch_dataloader
from dataset.utils import split_train_test_val_files
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn


warnings.filterwarnings(action='ignore')


random_seed = 424
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO: 128Hz로 Upsampling, 0.1 확률로 데이터에 gaussian noise


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--sampling_rate', default=100)
    parser.add_argument('--input_sec', default=30, type=int)
    parser.add_argument('--segment_size', default=35, type=int)
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', 'dataset', 'Sleep-EDFX-2018'), help='data path')
    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])

    # Train (for ContraWR)
    parser.add_argument('--train_epochs', default=100, type=int)
    parser.add_argument('--train_lr_rate', default=1e-6, type=float)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--channel_num', default=1, type=int)
    parser.add_argument('--num_block', default=12, type=int) # default 12
    parser.add_argument('--n_time_filters', default=5, type=int) # default 5
    parser.add_argument('--complexity_factor', default=2, type=float) # next step channel is 5*root2
    parser.add_argument('--class_num', default=5, type=int)
    parser.add_argument('--skip_connection', default=True, type=bool, help='presence of skip connection')
    parser.add_argument('--ensure_odd_conv_size', default=False, type=bool, help='If Ture, +1 to make even number to odd number')
    parser.add_argument('--conv_size', default=9, type=int, help='conv_size must be an odd number to accommodate the upsampling step in the decoder blocks')
    parser.add_argument('--apply_softmax', default=False, help='True => nn.NLLLoss, False => nn.CrossEntropyLoss')


    # Setting Checkpoint Path
    parser.add_argument('--print_point', default=20, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt',
                                                            'Sleep-EDFX', 'contrawr'), type=str)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = USleep(in_chans=self.args.channel_num,
                            sfreq=self.args.sampling_rate,
                            depth=self.args.num_block,
                            n_time_filters=self.args.n_time_filters,
                            complexity_factor=self.args.complexity_factor,
                            with_skip_connection=self.args.skip_connection,
                            n_classes=self.args.class_num,
                            input_size_s=self.args.input_sec,
                            time_conv_size_sec=self.args.conv_size,
                            ensure_odd_conv_size=self.args.ensure_odd_conv_size,
                            apply_softmax=self.args.apply_softmax
                            ).to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.train_paths, self.ft_paths, self.eval_paths = self.data_paths()
        self.writer_log = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'log.csv')
        self.device = device

    def train(self):
        print('K-Fold : {}/{}'.format(self.args.n_fold + 1, self.args.k_splits))
        train_dataloader = batch_dataloader(paths=self.train_paths, batch_size=self.args.train_batch_size, shuffle=True, segment_size=self.args.segment_size)
        eval_dataloader = batch_dataloader(paths=self.eval_paths, batch_size=self.args.train_batch_size, shuffle=False, segment_size=self.args.segment_size)
        print("Complete Loading DataSet")

        # Train (for U_SLEEP)
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_train_loss = []
            for batch in train_dataloader:
                x, y = batch # (b, channels, samples * segment), (b, segment) -> (b, 1, 105000), (b, 35)
                x, y = x.to(device), y.to(device)

                self.optimizer.zero_grad()
                pred = self.model(x) # (b, 5, 35) => (batch, class, segment)

                loss = self.criterion(pred, y) # logits shape needed: (b, class, D1, D2 ...), label shape needed: (b, D1, D2 ...)

                epoch_train_loss.append(float(loss.detach().cpu().item()))

                loss.backward()
                self.optimizer.step()

            epoch_train_loss = np.mean(epoch_train_loss)
            if (epoch + 1) % self.args.print_point == 0:
                # Print Log & Save Checkpoint Path
                epoch_total_pred, epoch_total_real = self.compute_evaluation(eval_dataloader) # (889, 35)
                epoch_total_pred, epoch_total_real = epoch_total_pred.flatten(), epoch_total_real.flatten() # (889*35,)

                # Calculation Metric (Acc & MF1)
                epoch_total_acc, epoch_total_mf1 = accuracy_score(y_true=epoch_total_real, y_pred=epoch_total_pred), \
                                             f1_score(y_true=epoch_total_real, y_pred=epoch_total_pred, average='macro')

                print('[Epoch] : {0:03d} \t '
                      '[Train Loss] => {1:.4f} \t '
                      '[Evaluation] => Acc: {2:.4f} MF1: {3:.4f}'.format(
                        epoch + 1, epoch_train_loss, epoch_total_acc, epoch_total_mf1))
                print(epoch_total_pred, end='\t')
                print(epoch_total_real)

            else:
                print('[Epoch] : {0:03d} \t '
                      '[Train Loss] => {1:.4f}'.format(epoch + 1, epoch_train_loss))


    def compute_evaluation(self, dataloader):
        self.model.eval()
        total_pred, total_real = [], []
        for batch in dataloader:
            with torch.no_grad():
                x, y = batch # (b, channels, samples * segment), (b, segment) -> (b, 1, 105000), (b, 35)
                x = x.to(device)
                pred = self.model(x) # (b, 5, 35) => (batch, class, segment)
                pred = pred.argmax(dim=1).detach().cpu().numpy() # (b, segment)
                real = y # (b, segment)
                total_pred.extend(pred)
                total_real.extend(real)
        total_pred, total_real = np.array(total_pred), np.array(total_real)
        return total_pred, total_real

    def data_paths(self):
        kf = split_train_test_val_files(base_path=self.args.base_path, n_splits=self.args.k_splits)
        paths = kf[self.args.n_fold]
        train_paths, ft_paths, eval_paths = paths['train_paths'], paths['ft_paths'], paths['eval_paths']
        return train_paths, ft_paths, eval_paths


if __name__ == '__main__':
    augments = get_args()
    print(f"CUDA Availablity: {torch.cuda.is_available()}")
    for n_fold in range(augments.k_splits):
        augments.n_fold = n_fold
        trainer = Trainer(augments)
        trainer.train()

