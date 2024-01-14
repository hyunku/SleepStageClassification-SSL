# -*- coding:utf-8 -*-
import os
import mne
import argparse
from experiments.ssl_sleep.multi_mae.model import *
from experiments.ssl_sleep.multi_mae import batch_dataloader
from dataset.utils import split_train_test_val_files
from collections import OrderedDict


warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--sampling_rate', default=100)
    parser.add_argument('--base_path', default=os.path.join('../../ssl', '..', '..', 'data', 'stage', 'Sleep-EDFX-2018'))
    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])

    # Train (for MAE)
    parser.add_argument('--train_epochs', default=300, type=int)
    parser.add_argument('--train_lr_rate', default=5e-4, type=float)
    parser.add_argument('--train_batch_size', default=512, type=int)

    # Masked Autoencoder Hyperparameter
    parser.add_argument('--band_range', default=OrderedDict({'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
                                                             'beta': (12, 15), 'gamma': (15, 30)}))
    parser.add_argument('--input_size', default=100 * 30, type=int)
    parser.add_argument('--patch_size', default=int(100 / 20), type=int)
    parser.add_argument('--emb_dim', default=192, type=int)
    parser.add_argument('--encoder_layer', default=12, type=int)
    parser.add_argument('--decoder_layer', default=4, type=int)
    parser.add_argument('--encoder_head', default=3, type=int)
    parser.add_argument('--decoder_head', default=3, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float)

    # Setting Checkpoint Path
    parser.add_argument('--print_point', default=10, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('../../ssl', '..', '..', 'ckpt',
                                                            'Sleep-EDFX', 'mae', 'BaseCNN'), type=str)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # self.model = MAE_ViT(input_size=args.input_size,
        #                      patch_size=args.patch_size,
        #                      emb_dim=args.emb_dim,
        #                      encoder_layer=args.encoder_layer, encoder_head=args.encoder_head,
        #                      decoder_layer=args.decoder_layer, decoder_head=args.decoder_head,
        #                      mask_ratio=args.mask_ratio).to(device)
        # self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.train_paths, self.ft_paths, self.eval_paths = self.data_paths()
        # self.writer_log = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'log.csv')

    def train(self):
        print('K-Fold : {}/{}'.format(self.args.n_fold + 1, self.args.k_splits))
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(paths=self.train_paths, band_range=self.args.band_range,
                                            sampling_rate=self.args.sampling_rate, batch_size=self.args.train_batch_size)

        # Train (for MAE)
        total_loss, total_bf_acc, total_bf_mf1, total_ft_acc, total_ft_mf1 = [], [], [], [], []
        for epoch in range(self.args.train_epochs):
            # self.model.train()

            epoch_train_loss = []
            for batch in train_dataloader.gather_async(num_async=5):
                print(batch.shape)

            exit()
        #         self.optimizer.zero_grad()
        #         signal = batch.to(device)
        #         pred_signal, mask = self.model(signal)
        #
        #         loss = torch.mean((pred_signal - torch.squeeze(signal)) ** 2 * mask) / self.args.mask_ratio
        #
        #         epoch_train_loss.append(float(loss.detach().cpu().item()))
        #         loss.backward()
        #         self.optimizer.step()
        #
        #     epoch_train_loss = np.mean(epoch_train_loss)
        #     if (epoch + 1) % self.args.print_point == 0 or epoch == 0:
        #         # Print Log & Save Checkpoint Path
        #         (epoch_bf_pred, epoch_bf_real), (epoch_ft_pred, epoch_ft_real) = self.compute_evaluation()
        #
        #         # Calculation Metric (Acc & MF1)
        #         epoch_bf_acc, epoch_bf_mf1 = accuracy_score(y_true=epoch_bf_real, y_pred=epoch_bf_pred), \
        #                                      f1_score(y_true=epoch_bf_real, y_pred=epoch_bf_pred, average='macro')
        #         epoch_ft_acc, epoch_ft_mf1 = accuracy_score(y_true=epoch_ft_real, y_pred=epoch_ft_pred),\
        #                                      f1_score(y_true=epoch_ft_real, y_pred=epoch_ft_pred, average='macro')
        #
        #         print('[Epoch] : {0:03d} \t '
        #               '[Train Loss] => {1:.4f} \t '
        #               '[Fine Tuning] => Acc: {2:.4f} MF1 {3:.4f} \t'
        #               '[Frozen Backbone] => Acc: {4:.4f} MF1: {5:.4f}'.format(
        #                 epoch + 1, epoch_train_loss, epoch_ft_acc, epoch_ft_mf1, epoch_bf_acc, epoch_bf_mf1))
        #
        #         self.save_ckpt(epoch, epoch_train_loss, epoch_bf_pred, epoch_bf_real, epoch_ft_pred, epoch_ft_real)
        #         total_loss.append(epoch_train_loss)
        #         total_bf_acc.append(epoch_bf_acc)
        #         total_bf_mf1.append(epoch_bf_mf1)
        #         total_ft_acc.append(epoch_ft_acc)
        #         total_ft_mf1.append(epoch_ft_mf1)
        #     else:
        #         print('[Epoch] : {0:03d} \t '
        #               '[Train Loss] => {1:.4f}'.format(epoch + 1, epoch_train_loss))
        #
        # epoch_labels = [i * self.args.print_point for i in range(len(total_loss))]
        # epoch_labels[0] = 1
        # df = pd.DataFrame({'epoch': epoch_labels,
        #                    'loss': total_loss,
        #                    'bf_acc': total_bf_acc, 'bf_mf1': total_bf_mf1,
        #                    'ft_acc': total_ft_acc, 'ft_mf1': total_ft_mf1})
        # df.to_csv(self.writer_log, index=False)
        ray.shutdown()

    # def save_ckpt(self, epoch, train_loss, bf_pred, bf_real, ft_pred, ft_real):
    #     if not os.path.exists(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model')):
    #         os.makedirs(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model'))
    #
    #     ckpt_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model', '{0:04d}.pth'.format(epoch+1))
    #     torch.save({
    #         'epoch': epoch+1,
    #         'backbone_name': 'MAE',
    #         'backbone_parameter': {
    #             'input_size': self.args.input_size, 'patch_size': self.args.patch_size, 'emb_dim': self.args.emb_dim,
    #             'encoder_layer': self.args.encoder_layer, 'encoder_head': self.args.encoder_head,
    #             'decoder_layer': self.args.decoder_layer, 'decoder_head': self.args.decoder_head,
    #             'mask_ratio': self.args.mask_ratio
    #         },
    #         'model_state': self.model.backbone.state_dict(),
    #         'hyperparameter': self.args.__dict__,
    #         'loss': train_loss,
    #         'result': {
    #             'frozen_backbone': {'real': bf_real, 'pred': bf_pred},
    #             'fine_tuning': {'real': ft_real, 'pred': ft_pred}
    #         },
    #         'paths': {'train_paths': self.train_paths, 'ft_paths': self.ft_paths, 'eval_paths': self.eval_paths}
    #     }, ckpt_path)
    #
    # def compute_evaluation(self):
    #     # 1. Backbone Frozen (train only fc)
    #     evaluation = Evaluation(backbone=self.model.backbone, device=device)
    #     bf_pred, bf_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
    #                                               eval_paths=self.eval_paths,
    #                                               frozen_layers=['conv1', 'conv2', 'conv3'])
    #     # 2. Fine Tuning (train last layer + fc)
    #     ft_pred, ft_real = evaluation.fine_tuning(ft_paths=self.ft_paths, eval_paths=self.eval_paths,
    #                                               frozen_layers=['conv1', 'conv2'])
    #     del evaluation
    #     return (bf_pred, bf_real), (ft_pred, ft_real)
    #
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

