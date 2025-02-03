import numpy as np
import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, max_t: int):
        super().__init__()
        self._conv1 =  nn.Conv2d(3, 32, 3, 1, padding='same')
        self._ln1 = nn.LayerNorm([32,16,16])
        self._te1 = nn.Embedding(max_t+1, 32)

        self._conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self._ln2 = nn.LayerNorm([64,8,8])
        self._te2 = nn.Embedding(max_t+1, 64)

        self._conv3 = nn.Conv2d(64, 128, 3, 1, padding='same')
        self._ln3 = nn.LayerNorm([128,4,4])
        self._te3 = nn.Embedding(max_t+1, 128)

        self._deconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self._ln4 = nn.LayerNorm([64,8,8])
        self._te4 = nn.Embedding(max_t+1, 64)

        self._deconv2 = nn.ConvTranspose2d(128, 32, 2, 2)
        self._ln5 = nn.LayerNorm([32,16,16])
        self._te5 = nn.Embedding(max_t+1, 32)

        self._deconv3 = nn.ConvTranspose2d(64, 3, 2, 2)
        self._ln6 = nn.LayerNorm([3,32,32])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x1 = nn.functional.max_pool2d(self._conv1(x), 2)  # Bx3x32x32 -> Bx32x16x16
        x1 = x1 + self._te1(t).view(1, 32, 1, 1)
        x1 = self._ln1(x1)
        x1 = nn.functional.relu(x1)
        x2 = nn.functional.max_pool2d(self._conv2(x1), 2)  # Bx32x16x16 -> Bx64x8x8
        x2 = x2 + self._te2(t).view(1, 64, 1, 1)
        x2 = self._ln2(x2)
        x2 = nn.functional.relu(x2)
        x3 = nn.functional.max_pool2d(self._conv3(x2), 2)  # Bx64x8x8 -> Bx128x4x4
        x3 = x3 + self._te3(t).view(1, 128, 1, 1)
        x3 = self._ln3(x3)
        x3 = nn.functional.relu(x3)
        y1 = self._ln4(self._deconv1(x3))  # Bx128x4x4 -> Bx64x8x8
        y1 = y1 + self._te4(t).view(1, 64, 1, 1)
        y1 = nn.functional.relu(y1)
        y2 = self._ln5(self._deconv2(torch.cat([y1, x2], dim=1)))  # Bx(64+64)x8x8 -> Bx32x16x16
        y2 = y2 + self._te5(t).view(1, 32, 1, 1)
        y2 = nn.functional.relu(y2)
        y3 = self._ln6(self._deconv3(torch.cat([y2, x1], dim=1)))  # Bx(32+32)x16x16 -> Bx3x32x32
        y3 = nn.functional.relu(y3)
        return y3

class DDPM(nn.Module):
    def __init__(self, max_t: int = 1000):
        super().__init__()
        self._unet = UNet(max_t)
        self._max_t = max_t

    def xt(self, x0: torch.Tensor, epsilon: torch.Tensor, t: int) -> torch.Tensor:
        """
        :param x0: real images
        :param epsilon: multivariate gaussian noise sampled from N(0,1)
        :param t: timestep \\in [0, T]
        :return: xt: the noisy version of x0 at timestep t, i.e.
                \\sqrt{\\pi_{s=1}^t \\alpha_s} x_0 + \\sqrt{1 - \\pi_{s=1}^t \\alpha_s} epsilon
        """

        alpha_bar_t = self.alpha_bar_t(t)
        return np.sqrt(alpha_bar_t) * x0 + np.sqrt(1-alpha_bar_t) * epsilon

    def forward(self, x0: torch.Tensor, epsilon: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x0: real images
        :param epsilon: multivariate gaussian noise sampled from N(0,1)
        :param t: timestep \\in [0, T]
        :return: predicted noise and the MSE loss

        Step 1: Sample x0, epsilon and a t
        Step 2 (this method): compute xt, the predicted noise using f(xt, t) and the MSE loss
        """

        assert x0.shape == epsilon.shape
        xt = self.xt(x0, epsilon, t)
        epsilon_pred = self._unet(xt, torch.tensor(t))
        loss = nn.functional.mse_loss(epsilon_pred, epsilon)
        return xt, epsilon_pred, loss

    @torch.no_grad()
    def beta_t(self, t: int) -> float:
        beta_min = 0.0001
        beta_max = 0.02
        return beta_min + (beta_max - beta_min) * t/self._max_t

    @torch.no_grad()
    def alpha_bar_t(self, t: int) -> float:
        prod = 1
        for s in range(t+1):
            prod *= 1 - self.beta_t(s)
        return prod

    @torch.no_grad()
    def sample(self, n: int = 16, save_every_n_steps: int = 100) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        xt = torch.randn(n, 3, 32, 32)
        saved = []
        for t in reversed(range(self._max_t)):
            z = torch.randn_like(xt) if t > 1 else 0
            alpha_t = 1 - self.beta_t(t)
            alpha_bar_t = self.alpha_bar_t(t)
            epsilon_pred = self._unet(xt, torch.tensor(t))
            sigma_t = np.sqrt(1-alpha_t)
            xt = (1/np.sqrt(alpha_t)) * (xt - ((1-alpha_t)/np.sqrt(1-alpha_bar_t)) * epsilon_pred) + z * sigma_t
            if t % save_every_n_steps == 0:
                saved.append((t, xt.clone()))
        return xt, saved
