# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
from models.utils import get_backbone_model, ProjectionHead
import torch.nn.functional as f
import sys
import einops


class MoCo(nn.Module): # Ver2, q: base encoder, k: momentum encoder + queue
    def __init__(self, backbone_name, backbone_parameter, projection_hidden, projection_size, temperature, queue_dim):
        super().__init__()
        self.projection_hidden = projection_hidden
        self.projection_size = projection_size

        self.encoder_q = get_backbone_model(model_name=backbone_name,
                                            parameters=backbone_parameter)
        self.encoder_k = get_backbone_model(model_name=backbone_name,
                                            parameters=backbone_parameter)
        self.projector_q = ProjectionHead(in_features=self.encoder_q.final_length,
                                          hidden_features=self.projection_hidden,
                                          out_features=self.projection_size,
                                          head_type='linear')
        self.projector_k = ProjectionHead(in_features=self.encoder_k.final_length,
                                          hidden_features=self.projection_hidden,
                                          out_features=self.projection_size,
                                          head_type='linear')
        self.K = queue_dim
        self.m = 0.999 # momentum
        self.T = temperature

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize -> 인코더값(encoder_q)을 momentum encoder(encoder_k) 에 copy 해주는 것
            param_k.requires_grad = False  # not update by gradient -> momentum encoder는 학습하지 않음.

        # create the queue
        self.register_buffer("queue", torch.randn(self.projection_size, self.K)) # queue 라는 변수를 만든다. 이 변수는 버퍼라서 gradient 계산을 하지 않는다. (feature dim(emb), queue차원)
        self.queue = nn.functional.normalize(self.queue, dim=0) # 출력차원 기준으로 normalize

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) # 큐의 어떤 자리에 enque, deque를 할지 결정하는 큐 포인터 생성

    def forward(self, x, mode='train'):
        if mode == 'train':
            x_q, x_k = x # x1, x2 shape : (256, 1, 3000) # (batch, height(eeg chan), width(time))
            z_q = self.encoder_q(x_q) # (256, 16256)
            z_q = self.projector_q(z_q) # q (batch(N), emb size(C)) (256, 128)
            q = f.normalize(z_q, dim=1) # normalize

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                z_k = self.encoder_k(x_k)  # keys: NxC
                z_k = self.projector_k(z_k)
                k = f.normalize(z_k, dim=1)

            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # q와 k의 내적
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            logits /= self.T

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            # calculate loss
            loss = nn.CrossEntropyLoss()(logits, labels)

            return loss

        elif mode == 'eval':
            feature = self.backbone(x)
            out = self.fc(feature)
            return out

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m) # momentum update

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # print(ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T  # dim x K # 큐의 포인터부터 배치사이즈만큼의 keys.T 를 큐에 넣는다 -> enque -> 덮어쓰기이므로 deque enque 동시에진행
        ptr = (ptr + batch_size) % self.K  # move pointer # 포인터 옮겨줌(배치사이즈 만큼의 인덱스 포인터)

        self.queue_ptr[0] = ptr