import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1d(nn.Module):
    def __init__(self, in_d, hid_d, out_d, stride, downsample=False):
        super().__init__()
        # 1x1 conv
        self.layer_1 = nn.Sequential(
            nn.Conv1d(in_d, hid_d, 1, 1, 0),
            nn.BatchNorm1d(hid_d),
            nn.ReLU(),
        )
        # 3x3 conv
        self.layer_2 = nn.Sequential(
            nn.Conv1d(hid_d, hid_d, 3, stride, 1),
            nn.BatchNorm1d(hid_d),
            nn.ReLU()
        )
        # 1x1 conv
        self.layer_3 = nn.Sequential(
            nn.Conv1d(hid_d, out_d, 1, 1, 0),
            nn.BatchNorm1d(out_d),
        )
        self.relu = nn.ReLU()

        if downsample:
            self.res_downsample = nn.Sequential(
                nn.Conv1d(in_d, out_d, 1, stride),
                nn.BatchNorm1d(out_d))
        else:
            self.res_downsample = None

    def forward(self, x):
        residual = x.clone()
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        if self.res_downsample:
            residual = self.res_downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNet1d(nn.Module):
    def __init__(self, in_d, n_layers, n_classes):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv1d(in_d, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.block_1 = self.make_block(64, 256, n_layers[0], 1)
        self.block_2 = self.make_block(256, 512, n_layers[1], 2)
        self.block_3 = self.make_block(512, 1024, n_layers[2], 2)
        self.block_4 = self.make_block(1024, 2048, n_layers[3], 2)

        self.out_layer = nn.Linear(2048, n_classes)

    def make_block(self, in_d, out_d, n_layers, stride):
        '''
        Creates one downsample block (stride 2)
        And the rest are plain resblocks
        '''
        self.layers = nn.ModuleList()
        self.layers.append(ResidualBlock1d(
            in_d, out_d//4, out_d, stride, downsample=True))
        for i in range(n_layers-1):
            self.layers.append(ResidualBlock1d(
                out_d, out_d//4, out_d, 1, downsample=False))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = F.avg_pool1d(x, x.shape[2])
        x = torch.flatten(x, 1)
        x = self.out_layer(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, model_name: str, n_classes: str, vocab_size: int, padding_idx: int):
        super().__init__()
        self.text_emb = nn.Embedding(vocab_size, 512, padding_idx=padding_idx)
        self.model = self._load_model(model_name, 1024)
        self.out_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=-1)
        )
        self.resnet = ResNet1d(1024, [2,2,2,2], 1024)

    def forward(self, text):
        x = self.text_emb(text)
        # x = x.permute(0, 2, 1).contiguous()
        x, _ = self.model(x)
        x = self.resnet(x.permute(0,2,1).contiguous())
        x = self.out_layer(x)
        return x
        
    def _load_model(self, model_name, n_classes):
        if model_name == 'resnet18':
            return ResNet1d(512, [2, 2, 2, 2], n_classes)
        elif model_name == 'resnet50':
            return ResNet1d(512, [3, 4, 6, 3], n_classes)
        elif model_name == 'resnet101':
            return ResNet1d(512, [3, 4, 23, 3], n_classes)
        elif model_name == 'resnet152':
            return ResNet1d(512, [3, 8, 36, 3], n_classes)
        elif model_name == 'lstm':
            return nn.LSTM(512, 512, batch_first=True, num_layers=3, bidirectional=True,)
        else:
            raise NotImplementedError(f'Model: {model_name} not found')


def load_model(model_name, n_classes, vocab_size, padding_idx):
    return TextCNN(model_name, n_classes, vocab_size, padding_idx)
  
