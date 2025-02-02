import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv1 =  nn.Conv2d(3, 32, 3, 1, padding='same')
        self._ln1 = nn.LayerNorm([32,16,16])
        self._conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self._ln2 = nn.LayerNorm([64,8,8])
        self._conv3 = nn.Conv2d(64, 128, 3, 1, padding='same')
        self._ln3 = nn.LayerNorm([128,4,4])
        self._deconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self._ln4 = nn.LayerNorm([64,8,8])
        self._deconv2 = nn.ConvTranspose2d(128, 32, 2, 2)
        self._ln5 = nn.LayerNorm([32,16,16])
        self._deconv3 = nn.ConvTranspose2d(64, 3, 2, 2)
        self._ln6 = nn.LayerNorm([3,32,32])

    def forward(self, x: torch.Tensor):
        x1 = nn.functional.max_pool2d(self._conv1(x), 2)  # Bx3x32x32 -> Bx32x16x16
        x1 = self._ln1(x1)
        x1 = nn.functional.relu(x1)
        x2 = nn.functional.max_pool2d(self._conv2(x1), 2)  # Bx32x16x16 -> Bx64x8x8
        x2 = self._ln2(x2)
        x2 = nn.functional.relu(x2)
        x3 = nn.functional.max_pool2d(self._conv3(x2), 2)  # Bx64x8x8 -> Bx128x4x4
        x3 = self._ln3(x3)
        x3 = nn.functional.relu(x3)
        y1 = self._ln4(self._deconv1(x3))  # Bx128x4x4 -> Bx64x8x8
        y1 = nn.functional.relu(y1)
        y2 = self._ln5(self._deconv2(torch.cat([y1, x2], dim=1)))  # Bx(64+64)x8x8 -> Bx32x16x16
        y2 = nn.functional.relu(y2)
        y3 = self._ln6(self._deconv3(torch.cat([y2, x1], dim=1)))  # Bx(32+32)x16x16 -> Bx3x32x32
        y3 = nn.functional.relu(y3)
        return y3


class DDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self._unet = UNet()

    def forward(self):
        pass
