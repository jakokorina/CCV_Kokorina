import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, d=5):
        super().__init__()
        self.d = d
        self.enter = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU()
        )
        self.core = []
        for i in range(d):
            self.core.append(nn.Conv2d(64, 64, 3, padding=1))
            self.core.append(nn.BatchNorm2d(64))
            self.core.append(nn.ReLU())
        self.out = nn.Conv2d(64, 3, 3, padding=1)
        self.model = nn.Sequential(self.enter, *self.core, self.out)

    def forward(self, x):
        return self.model(x)
