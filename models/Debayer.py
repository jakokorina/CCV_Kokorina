import torch.nn as nn


class DebayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 12, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.model(x)
        return nn.functional.pixel_shuffle(out, 2)
