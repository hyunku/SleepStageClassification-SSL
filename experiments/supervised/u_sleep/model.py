import numpy as np
import torch
from torch import nn



def _crop_tensors_to_match(x1, x2, axis=-1):
    dim_cropped = min(x1.shape[axis], x2.shape[axis])

    x1_cropped = torch.index_select(
        x1, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    x2_cropped = torch.index_select(
        x2, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    return x1_cropped, x2_cropped


class _EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=2, # filter(논문에서 언급한 스타일)
                 kernel_size=9, # kernel
                 downsample=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample

        self.block_prepool = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )

        self.pad = nn.ConstantPad1d(padding=1, value=0)
        self.maxpool = nn.MaxPool1d(
            kernel_size=self.downsample, stride=self.downsample)

    def forward(self, x):
        x = self.block_prepool(x)
        residual = x
        if x.shape[-1] % 2: # 홀수일 때
            x = self.pad(x) # 남는 1칸에 패딩
        x = self.maxpool(x)
        return x, residual


class _DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 upsample=2,
                 with_skip_connection=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection

        self.block_preskip = nn.Sequential(
            nn.Upsample(scale_factor=upsample),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=2,
                      padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.block_postskip = nn.Sequential(
            nn.Conv1d(
                in_channels=(
                    2 * out_channels if with_skip_connection else out_channels),
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )

    def forward(self, x, residual):
        x = self.block_preskip(x)
        if self.with_skip_connection:
            x, residual = _crop_tensors_to_match(x, residual, axis=-1)  # in case of mismatch
            x = torch.cat([x, residual], axis=1)  # (B, 2 * C, T)
        x = self.block_postskip(x)
        return x


class USleep(nn.Module):
    def __init__(self,
                 in_chans=1,
                 sfreq=100,
                 depth=12,
                 n_time_filters=5, # 채널(filter) 조정 파라미터 초기값
                 complexity_factor=2, # 다음 채널(filter) 는 루트2 씩 곱한 값임
                 with_skip_connection=True,
                 n_classes=5,
                 input_size_s=30,
                 time_conv_size_sec=9, # enc, dec에서의 컨볼루션 kernel size
                 ensure_odd_conv_size=False,
                 apply_softmax=False
                 ):
        super().__init__()

        time_conv_size_s = time_conv_size_sec / sfreq
        self.in_chans = in_chans
        max_pool_size = 2  # Hardcoded to avoid dimensional errors
        time_conv_size = np.round(time_conv_size_s * sfreq).astype(int)
        if time_conv_size % 2 == 0:
            if ensure_odd_conv_size:
                time_conv_size += 1
            else:
                raise ValueError(
                    'time_conv_size must be an odd number to accomodate the '
                    'upsampling step in the decoder blocks.')

        # Convert between units: seconds to time-points (at sfreq)
        input_size = np.ceil(input_size_s * sfreq).astype(int) # 3000 (30sec*100Hz) -> sample size

        channels = [in_chans] # [1, 7, 9, 12, 16, 22, 31, 43, 60, 84, 118, 166, 234, 330]
        n_filters = n_time_filters
        for _ in range(depth + 1):
            channels.append(int(n_filters * np.sqrt(complexity_factor)))
            n_filters = int(n_filters * np.sqrt(2))
        self.channels = channels

        # Instantiate encoder => [7, 9, 12, 16, 22, 31, 43, 60, 84, 118, 166, 234] output channel 234
        encoder = list()
        for idx in range(depth):
            encoder += [
                _EncoderBlock(in_channels=channels[idx],
                              out_channels=channels[idx + 1],
                              kernel_size=time_conv_size,
                              downsample=max_pool_size)
            ]
        self.encoder = nn.Sequential(*encoder)

        # Instantiate bottom (channels increase, temporal dim stays the same)
        self.bottom = nn.Sequential(
                    nn.Conv1d(in_channels=channels[-2],
                              out_channels=channels[-1],
                              kernel_size=time_conv_size,
                              padding=(time_conv_size - 1) // 2),  # preserves dimension
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=channels[-1]),
                )

        # Instantiate decoder
        decoder = list()
        channels_reverse = channels[::-1]
        for idx in range(depth):
            decoder += [
                _DecoderBlock(in_channels=channels_reverse[idx],
                              out_channels=channels_reverse[idx + 1],
                              kernel_size=time_conv_size,
                              upsample=max_pool_size,
                              with_skip_connection=with_skip_connection)
            ]
        self.decoder = nn.Sequential(*decoder)

        # self.clf = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=channels[1],
        #         out_channels=channels[1],
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     ),                         # (b, c, s*seg*w)
        #     nn.Tanh(),
        #     nn.AvgPool1d(input_size),  # (b, c, s*seg*w) -> (b, c, seg*w)
        #     nn.Conv1d(
        #         in_channels=channels[1],
        #         out_channels=n_classes,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     ),                         # (b, n_classes, seg*w)
        #     nn.ELU(),
        #     nn.Conv1d(
        #         in_channels=n_classes,
        #         out_channels=n_classes,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     ),
        #     nn.Softmax(dim=1) if apply_softmax else nn.Identity(), # (b, n_classes, seg*w)
        # )

        self.clf = nn.Sequential( # (b, c, s*seg) = (b, 7, 3000*35) # 논문 보고 직접 구현한 version
            nn.AvgPool1d(input_size), # (b, c, seg) => 샘플(3000=i)에 대해서 kernel, stride=i 인 각 채널별 avgpool 진행 => (b, 7, 35)
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=n_classes,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Softmax(dim=1) if apply_softmax else nn.Identity(), # (b, n_classes, seg) => 논문에서 segment를 T, 클래스를 K 로 표현하여 T X K matrix로 말함
        )

    def forward(self, x): # (batch, window(sequence), channel(feature), sample)
        # reshape input
        if x.ndim == 4:  # input x has shape (b, w, c, s)
            x = x.permute(0, 2, 1, 3)  # (b, c, w, s)
            x = x.flatten(start_dim=2)  # (b, c, w * s) -> (b, c, ws)

        # encoder
        residuals = []
        for down in self.encoder:
            x, res = down(x)
            residuals.append(res)

        # bottom
        x = self.bottom(x) # encoder output: (b, 234, 26) -> bottom output: (b, 330, 26)

        # decoder
        residuals = residuals[::-1]  # flip order
        for up, res in zip(self.decoder, residuals): # (b, c, s*seg*w) => (b, 7, 105000)
            x = up(x, res)

        # classifier
        y_pred = self.clf(x)        # (b, n_classes, n_segment*window)

        if y_pred.shape[-1] == 1:  # n_segment = 1 인 경우
            y_pred = y_pred[:, :, 0] # (b, n_class, 1) -> (b, n_class)

        return y_pred # (b, class, segmentation) -> window 어차피 1임

def print_shape(module, input, output):
    print(f'{module.__class__.__name__} ======> input shape: {input[0].shape}, output shape: {output.shape}')

if __name__ == '__main__': # (batch, window(sequence), channel(feature), sample) => (b, w, c, s)
    m = USleep()
    # for layer in m.clf:
    #     layer.register_forward_hook(print_shape)
    d = torch.randn(size=(64, 1, 105000))  # b, c, s*seg (3000*35)
    # y = torch.randint(size=(64, 35)) # b, seg(35)
    # y = torch.tensor(y, dtype=torch.long)
    pred = m(d)
    print(pred)
    # print(y.shape, pred.shape) # y: (b,35), pred: (b,5,35) (b, class, seg)
    #
    # batch_size = pred.size(0)
    # segment = pred.size(2)
    #
    # pred = pred.view(batch_size * segment, -1)  # Shape: (batch * 35, 5)
    # target = y.view(-1)  # Shape: (batch * 35)
    # print(pred.shape, target.shape)
    #
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(pred, target)
    # print(loss)

