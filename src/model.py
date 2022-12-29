import torch
import torch.nn as nn


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

