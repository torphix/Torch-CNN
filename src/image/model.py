import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaCNN(torch.nn.Module):
    def __init__(self, n_layers, n_classes):
        super().__init__()

        self.layers = nn.ModuleList()
        # In Layer
        self.layers.append(nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ))

        # Intermediate Layers
        in_d = 32
        for i in range(n_layers-1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_d, in_d*2, 3, 1, 0),
                nn.BatchNorm2d(in_d*2),
                nn.ReLU(),
            ))
            if in_d < 1024:
                in_d *= 2

        # Out Layer
        self.layers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_d, n_classes),
        ))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)



class ResidualBlock(nn.Module):
    def __init__(self, in_d, hid_d, out_d, stride, downsample=False):
        super().__init__()
        # 1x1 conv
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_d, hid_d, (1,1), 1, 0),
            nn.BatchNorm2d(hid_d),
            nn.ReLU(),
        )
        # 3x3 conv
        self.layer_2 = nn.Sequential(
            nn.Conv2d(hid_d, hid_d, (3,3), stride, 1),
            nn.BatchNorm2d(hid_d),
            nn.ReLU()
        )
        # 1x1 conv
        self.layer_3 = nn.Sequential(
            nn.Conv2d(hid_d, out_d, (1,1), 1, 0),
            nn.BatchNorm2d(out_d),
        )
        self.relu = nn.ReLU()

        if downsample:
            self.res_downsample = nn.Sequential(
                nn.Conv2d(in_d, out_d, (1,1), stride),
                nn.BatchNorm2d(out_d))
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

class ResNet(nn.Module):
    def __init__(self, in_d, n_layers, n_classes):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_d, 64, (7, 7), 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

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
        self.layers.append(ResidualBlock(in_d, out_d//4, out_d, stride, downsample=True))
        for i in range(n_layers-1):
            self.layers.append(ResidualBlock(out_d, out_d//4, out_d, 1, downsample=False))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = F.avg_pool2d(x, x.shape[2])
        x = torch.flatten(x, 1)
        x = self.out_layer(x)
        return x


def load_model(model_name, n_classes, model_config=None):
    if model_name == 'vanilla_cnn':
        return VanillaCNN(**model_config, n_classes=n_classes)
    elif model_name == 'resnet18':
        return ResNet(3, [2, 2, 2, 2], n_classes)
    elif model_name == 'resnet50':
        return ResNet(3, [3, 4, 6, 3], n_classes)
    elif model_name == 'resnet101':
        return ResNet(3, [3, 4, 23, 3], n_classes)
    elif model_name == 'resnet152':
        return ResNet(3, [3, 8, 36, 3], n_classes)
    else:
        raise NotImplementedError(f'Model: {model_name} not found')