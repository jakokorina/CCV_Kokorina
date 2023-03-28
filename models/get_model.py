import torch.nn as nn


def get_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=(3, 3), padding='same'),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=(3, 3), padding='same'),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=(3, 3), padding='same'),
        nn.ReLU(),
        nn.Conv2d(16, 3, kernel_size=(3, 3), padding='same'),
    )
