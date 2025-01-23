import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class DDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self._unet = UNet()

    def forward(self):
        pass
