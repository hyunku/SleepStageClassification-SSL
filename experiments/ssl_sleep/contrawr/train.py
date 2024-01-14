# -*- coding:utf-8 -*-
import os
import mne
import argparse
import pandas as pd
import torch.optim as opt

from experiments.ssl_sleep.contrawr.model import ContraWR
from models.utils import get_backbone_parameter
from experiments.evaluation import Evaluation
from experiments.ssl_sleep.contrawr.data_loader import *
from experiments.ssl_sleep.contrawr.data_loader import batch_dataloader
from dataset.utils import split_train_test_val_files
from models.loss import TripletLoss
from sklearn.metrics import accuracy_score, f1_score


warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--sampling_rate', default=100)
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', 'data', 'stage', 'Sleep-EDFX-2018'))
    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])

    # Train (for ContraWR)
    parser.add_argument('--train_epochs', default=100, type=int)
    parser.add_argument('--train_lr_rate', default=2e-4, type=float)
    parser.add_argument('--train_batch_size', default=512, type=int)
    parser.add_argument('--temperature', default=0.2, type=float, help='used for triplet loss, default is 0.07')
    parser.add_argument('--margin', default=0.5, type=float, help='used for triplet loss, semi hard negatives margin')
    parser.add_argument('--sigma', default=2.0, type=float, help='used for triplet loss, standard deviation')
    parser.add_argument('--ema_momentum', default=0.9995, type=float, help='moving average coefficient')
    parser.add_argument('--backbone_name', default='CNNEncoder2D_SLEEP', type=str)
    parser.add_argument('--backbone_parameter', default=get_backbone_parameter(model_name='CNNEncoder2D_SLEEP',
                                                                               sampling_rate=100))

    # Setting Data Augmentation
    parser.add_argument('--augmentations', default=[('random_permutation', 0.85),
                                                    ('random_bandpass_filter', 0.85),
                                                    ('random_temporal_cutout', 0.85),
                                                    ('random_crop', 0.85)])
    parser.add_argument('--projection_hidden', default=512, type=int)
    parser.add_argument('--projection_size', default=256, type=int)

    # Setting Checkpoint Path
    parser.add_argument('--print_point', default=5, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt',
                                                            'Sleep-EDFX', 'contrawr'), type=str)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = ContraWR(backbone_name=self.args.backbone_name,
                              backbone_parameter=self.args.backbone_parameter,
                              projection_hidden=self.args.projection_hidden,
                              projection_size=self.args.projection_size,
                              ema_momentum=self.args.ema_momentum).to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.criterion = TripletLoss(temperature=self.args.temperature,
                                     margin=self.args.margin,
                                     sigma=self.args.sigma).to(device)
        self.train_paths, self.ft_paths, self.eval_paths = self.data_paths()
        self.writer_log = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'log.csv')

    def train(self):
        print('K-Fold : {}/{}'.format(self.args.n_fold + 1, self.args.k_splits))
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(paths=self.train_paths, batch_size=self.args.train_batch_size,
                                            augmentations=self.args.augmentations,
                                            freqs=self.args.sampling_rate)
        # Train (for ContraWR)
        total_loss, total_bf_acc, total_bf_mf1, total_ft_acc, total_ft_mf1 = [], [], [], [], []
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_train_loss = []
            for batch in train_dataloader.gather_async(num_async=5):
                x1, x2 = batch
                x1, x2 = x1.to(device), x2.to(device)
                z_i, z_j = self.model((x1, x2))
                loss = self.criterion(z_i, z_j)

                epoch_train_loss.append(float(loss.detach().cpu().item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_train_loss = np.mean(epoch_train_loss)
            if (epoch + 1) % self.args.print_point == 0 or epoch == 0:
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

                self.save_ckpt(epoch, epoch_train_loss, epoch_bf_pred, epoch_bf_real, epoch_ft_pred, epoch_ft_real)
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
            'backbone_name': self.args.backbone_name,
            'backbone_parameter': self.args.backbone_parameter,
            'model_state': self.model.encoder_q.backbone.state_dict(),
            'hyperparameter': self.args.__dict__,
            'loss': train_loss,
            'result': {
                'frozen_backbone': {'real': bf_real, 'pred': bf_pred},
                'fine_tuning': {'real': ft_real, 'pred': ft_pred}
            },
            'paths': {'train_paths': self.train_paths, 'ft_paths': self.ft_paths, 'eval_paths': self.eval_paths}
        }, ckpt_path)

    def compute_evaluation(self):
        # 1. Backbone Frozen (train only fc)
        evaluation = Evaluation(backbone=self.model.encoder_q.backbone, device=device)
        bf_pred, bf_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=['conv1', 'conv2', 'conv3', 'conv4'])
        # 2. Fine Tuning (train last layer + fc)
        ft_pred, ft_real = evaluation.fine_tuning(ft_paths=self.ft_paths, eval_paths=self.eval_paths,
                                                  frozen_layers=['conv1', 'conv2'])
        del evaluation
        return (bf_pred, bf_real), (ft_pred, ft_real)

    def data_paths(self):
        kf = split_train_test_val_files(base_path=self.args.base_path, n_splits=self.args.k_splits)
        paths = kf[self.args.n_fold]
        train_paths, ft_paths, eval_paths = paths['train_paths'], paths['ft_paths'], paths['eval_paths']
        return train_paths, ft_paths, eval_paths


if __name__ == '__main__':
    augments = get_args()
    for n_fold in range(augments.k_splits):
        augments.n_fold = n_fold
        trainer = Trainer(augments)
        trainer.train()

