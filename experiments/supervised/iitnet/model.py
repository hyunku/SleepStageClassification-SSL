import torch.nn as nn
import torch


# TODO: 논문환경: preprocess : X, dataset: sleep edfx, channel: "Fpz-Cz" 단일, lr=0.005
# TODO: 논문언급 안되었지만 실험에 사용된 모델 파라미터 - rnn hidden dim 128, num_layers: 50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x) # (b, 16, 750)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # (b, 16, 750)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # (b, 16*4, 750)
        out = self.bn3(out)

        if self.downsample is not None: # 차원 안맞으면 downsample layer 추가해서 차원 맞춰줌(16 -> 16*4)
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Main neural network class, IITNet
class IITNetBackBone(nn.Module):
    def __init__(self):
        super(IITNetBackBone, self).__init__()

        block = Bottleneck

        # Initial configuration
        self.inplanes = 16
        self.layers = [3, 4, 6, 3]

        # Initial layer # conv 연산 공식: O = [I - F + 2P / S] + 1 => [(3000 - 7 + 6) / 2] + 1  = 1500 (대괄호는 버림연산)
        self.initial_layer = nn.Sequential(  # 피쳐맵 사이즈 유지 - 1. padding='same', 2. F = 2P + 1
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False), # (b, 1, 3000) -> (b, 16, 1500)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) # 보통 (커널, 패딩) 조합이 (3,1), (5,2), (7,3), (9,4) 면 차원 유지되서 stride수만큼 나눈만큼 나옴 # (b, 16, 750)

        # Building layers using Bottleneck blocks
        self.layer1 = self._make_layer(block, 16, self.layers[0], stride=1, first=True)
        self.layer2 = self._make_layer(block, 16, self.layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, self.layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, self.layers[3], stride=2)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, first=False):
        # Create layers of blocks
        downsample = None
        if (stride != 1 and first is False) or self.inplanes != planes * block.expansion: # 입력과 출력 차원이 다를 때 downsample layer 추가해줘서 차원 맞춰줌
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), # 채널 16 -> 16*4
                nn.BatchNorm1d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x) # (b, 1, 3000) => (b, 16, 750) # batch, features, seq_len
        x = self.layer1(x) # (b, 64, 750)
        x = self.layer2(x) # (b, 64, 375)
        x = self.layer3(self.maxpool(x)) # (b, 128, 94)
        x = self.layer4(x) # (b, 128, 47) batch, features, seq_len

        return x


class PlainRNN(nn.Module):
    def __init__(self, num_classes=5, input_dim=128, hidden_dim=128, num_layers=50):
        super(PlainRNN, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def init_hidden(self, x):
        h0 = torch.zeros((self.num_layers * 2, x.size(0), self.hidden_dim))
        c0 = torch.zeros((self.num_layers * 2, x.size(0), self.hidden_dim))

        return h0, c0

    def forward(self, x):
        hidden = self.init_hidden(x)
        rnn_output, hidden = self.rnn(x, hidden)

        output_f = rnn_output[:, -1, :self.hidden_dim]  # Forward direction
        output_b = rnn_output[:, 0, self.hidden_dim:]  # Backward direction
        output = torch.cat((output_f, output_b), dim=1)

        output = self.fc(output)

        return output

class IITNet(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=128, num_layers=50):
        super(IITNet, self).__init__()

        self.backbone = IITNetBackBone()
        self.clf = PlainRNN(num_classes=num_classes,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers)

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0,2,1) # b, feature, seq_len -> b, seq_len, feature
        x = self.clf(x)
        return x




if __name__ == '__main__':
    m = IITNet()
    # for layer in m.clf:
    #     layer.register_forward_hook(print_shape)
    d = torch.randn(size=(64, 1, 3000))  # b, c, s*seg (3000*35)
    # m = PlainRNN()
    # d = torch.randn(size=(64, 128, 47)) # embedded feature
    pred = m(d)
    print(pred.shape)