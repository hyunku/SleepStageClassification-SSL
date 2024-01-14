# -*- coding:utf-8 -*-
import sys
sys.path.append('../../../../EEG_Self_Supervised_Learning/experiments/ssl')

import torch
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from models.utils import get_backbone_model, LinearLayer


class DeepCluster(nn.Module):
    def __init__(self, backbone_name, backbone_parameter,
                 pca_n_components, cluster_classes):
        super().__init__()
        self.backbone = get_backbone_model(model_name=backbone_name,
                                           parameters=backbone_parameter)
        self.cluster = GeneratorPseudoLabel(n_components=pca_n_components,
                                            classes=cluster_classes)
        self.fc = nn.Linear(in_features=self.backbone.final_length,
                            out_features=cluster_classes)

    def forward(self, x, mode='pseudo_labels'):
        if mode == 'pseudo_labels':
            # Create Pseudo Label
            with torch.no_grad():
                self.backbone.eval()
                conv_out = self.backbone(x)
                pseudo_label = self.cluster.train(conv_out)
            return conv_out, pseudo_label
        elif mode == 'eval':
            self.backbone.train()
            self.fc.train()
            conv_out = self.backbone(x)
            out = self.fc(conv_out)
            return out


class GeneratorPseudoLabel(object):
    def __init__(self, n_components, classes):
        self.n_components = n_components
        self.classes = classes

    def init_model(self):
        pca = PCA(n_components=self.n_components)
        kmeans = KMeans(n_clusters=self.classes)
        return pca, kmeans

    def train(self, x):
        device = x.get_device()
        x = x.cpu().detach().numpy()

        pca, kmeans = self.init_model()

        # PCA-reducing, whitening and L2-normalization
        x_pca = pca.fit_transform(x)
        norm = np.linalg.norm(x_pca, axis=1)
        x_l2 = x_pca / norm[:, np.newaxis]

        # Cluster the data (for K-Means)
        kmeans.fit(x_l2)
        pseudo_label = kmeans.labels_
        pseudo_label = torch.tensor(pseudo_label, dtype=torch.long).to(device)
        return pseudo_label
