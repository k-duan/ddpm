import numpy as np
import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, max_t: int):
        super().__init__()
        self._conv1 =  nn.Conv2d(3, 64, 3, 1, padding='same')
        self._conv1a = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._ln1 = nn.LayerNorm([64, 32, 32])
        self._ln1a = nn.LayerNorm([64, 32, 32])
        self._te1 = nn.Embedding(max_t+1, 64 * 16 * 16)

        self._conv2 = nn.Conv2d(64, 128, 3, 1, padding='same')
        self._conv2a = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._ln2 = nn.LayerNorm([128, 16, 16])
        self._ln2a = nn.LayerNorm([128, 16, 16])
        # self._te2 = nn.Embedding(max_t+1, 128)

        self._conv3 = nn.Conv2d(128, 256, 3, 1, padding='same')
        self._conv3a = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._ln3 = nn.LayerNorm([256, 8, 8])
        self._ln3a = nn.LayerNorm([256, 8, 8])
        #  self._te3 = nn.Embedding(max_t+1, 256)

        self._convb = nn.Conv2d(256, 512, 3, 1, padding='same')
        self._lnb = nn.LayerNorm([512, 4, 4])
        # self._teb = nn.Embedding(max_t+1, 512)

        self._deconv1 = nn.ConvTranspose2d(512+256, 256, 2, 2)
        self._deconv1a = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._ln4 = nn.LayerNorm([256, 8, 8])
        self._ln4a = nn.LayerNorm([256, 8, 8])
        # self._te4 = nn.Embedding(max_t+1, 256)

        self._deconv2 = nn.ConvTranspose2d(256+128, 128, 2, 2)
        self._deconv2a = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._ln5 = nn.LayerNorm([128, 16, 16])
        self._ln5a = nn.LayerNorm([128, 16, 16])
        # self._te5 = nn.Embedding(max_t+1, 128)

        self._deconv3 = nn.ConvTranspose2d(128+64, 64, 2, 2)
        self._deconv3a = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._ln6 = nn.LayerNorm([64, 32, 32])
        self._ln6a = nn.LayerNorm([64, 32, 32])
        # self._te6 = nn.Embedding(max_t + 1, 64)

        self._final_conv = nn.Conv2d(64, 3, 3, 1, padding='same')

    def forward(self, x: torch.Tensor, t: torch.Tensor, pos_emb: bool = True):
        x1 = self._conv1(x)
        x1 = self._ln1(x1)
        x1 = nn.functional.relu(x1)
        x1 = self._conv1a(x1)
        x1 = self._ln1a(x1)
        x1 = nn.functional.relu(x1)
        x1 = nn.functional.max_pool2d(x1, 2)  # Bx3x32x32 -> Bx64x16x16
        if pos_emb:
            x1 = x1 + self._te1(t).view(x.size(0), 64, 16, 16)

        x2 = self._conv2(x1)
        x2 = self._ln2(x2)
        x2 = nn.functional.relu(x2)
        x2 = self._conv2a(x2)
        x2 = self._ln2a(x2)
        x2 = nn.functional.relu(x2)
        x2 = nn.functional.max_pool2d(x2, 2)  # Bx64x16x16 -> Bx128x8x8
        # if pos_emb:
        #     x2 = x2 + self._te2(t).view(1, 128, 1, 1)

        x3 = self._conv3(x2)
        x3 = self._ln3(x3)
        x3 = nn.functional.relu(x3)
        x3 = self._conv3a(x3)
        x3 = self._ln3a(x3)
        x3 = nn.functional.relu(x3)
        x3 = nn.functional.max_pool2d(x3, 2)  # Bx128x8x8 -> Bx256x4x4
        # if pos_emb:
        #     x3 = x3 + self._te3(t).view(1, 256, 1, 1)

        x4 = self._convb(x3)
        x4 = self._lnb(x4)
        x4 = nn.functional.relu(x4)
        # if pos_emb:
        #     x4 = x4 + self._teb(t).view(1, 512, 1, 1)

        y1 = self._deconv1(torch.cat([x4, x3], dim=1))  # Bx(512+256)x4x4 -> Bx256x8x8
        y1 = self._ln4(y1)
        y1 = nn.functional.relu(y1)
        y1 = self._deconv1a(y1)
        y1 = self._ln4a(y1)
        y1 = nn.functional.relu(y1)
        # if pos_emb:
        #     y1 = y1 + self._te4(t).view(1, 256, 1, 1)

        y2 = self._deconv2(torch.cat([y1, x2], dim=1))  # Bx(256+128)x8x8 -> Bx128x16x16
        y2 = self._ln5(y2)
        y2 = nn.functional.relu(y2)
        y2 = self._deconv2a(y2)
        y2 = self._ln5a(y2)
        y2 = nn.functional.relu(y2)
        # if pos_emb:
        #     y2 = y2 + self._te5(t).view(1, 128, 1, 1)

        y3 = self._deconv3(torch.cat([y2, x1], dim=1))  # Bx(128+64)x16x16 -> Bx64x32x32
        y3 = self._ln6(y3)
        y3 = nn.functional.relu(y3)
        y3 = self._deconv3a(y3)
        y3 = self._ln6a(y3)
        y3 = nn.functional.relu(y3)
        # if pos_emb:
        #     y3 = y3 + self._te6(t).view(1, 64, 1, 1)

        y = self._final_conv(y3)
        y = torch.nn.functional.tanh(y)
        return y

class DDPM(nn.Module):
    def __init__(self, max_t: int = 1000):
        super().__init__()
        self._unet = UNet(max_t)
        self._max_t = max_t

    def xt(self, x0: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x0: real images, (B,C,H,W)
        :param epsilon: multivariate gaussian noise sampled from N(0,1), (B,C,H,W)
        :param t: timestep \\in [0, T], (B,)
        :return: xt: the noisy version of x0 at timestep t, i.e.
                \\sqrt{\\pi_{s=1}^t \\alpha_s} x_0 + \\sqrt{1 - \\pi_{s=1}^t \\alpha_s} epsilon
        """

        bs = x0.size(0)
        alpha_bar_t = self.alpha_bar_t(t)
        return torch.sqrt(alpha_bar_t).view(bs, 1, 1, 1) * x0 + torch.sqrt(1-alpha_bar_t).view(bs, 1, 1, 1) * epsilon

    def forward(self, x0: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x0: real images, (B,C,H,W)
        :param epsilon: multivariate gaussian noise sampled from N(0,1), (B,C,H,W)
        :param t: timestep \\in [0, T], (B,)
        :return: predicted noise and the MSE loss

        Step 1: Sample x0, epsilon and a t
        Step 2 (this method): compute xt, the predicted noise using f(xt, t) and the MSE loss
        """

        assert x0.shape == epsilon.shape
        xt = self.xt(x0, epsilon, t)
        epsilon_pred = self._unet(xt, t)
        loss = nn.functional.mse_loss(epsilon_pred, epsilon)
        return xt, epsilon_pred, loss

    @torch.no_grad()
    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        beta_min = 0.0001
        beta_max = 0.02
        beta_schedule = torch.linspace(beta_min, beta_max, self._max_t)
        return beta_schedule[t]

    @torch.no_grad()
    def alpha_bar_t(self, t: torch.Tensor) -> torch.Tensor:
        beta_min = 0.0001
        beta_max = 0.02
        beta_schedule = torch.linspace(beta_min, beta_max, self._max_t)
        prods = torch.ones_like(beta_schedule)
        for i in range(self._max_t):
            prods[i] = prods[i-1] * (1-beta_schedule[i]) if i > 0 else 1 - beta_schedule[0]
        return prods[t]

    @torch.no_grad()
    def sample(self, n: int = 16, save_every_n_steps: int = 100) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        xt = torch.randn(n, 3, 32, 32)
        saved = [(self._max_t, xt.clone())]
        for t in reversed(range(1, self._max_t)):
            ts = torch.full((n,), t)
            z = torch.randn_like(xt) if t > 1 else 0
            alpha_t = 1 - self.beta_t(ts)
            alpha_bar_t = self.alpha_bar_t(ts)
            epsilon_pred = self._unet(xt, ts)
            sigma_t = np.sqrt(1-alpha_t)
            xt = (1/np.sqrt(alpha_t)) * (xt - ((1-alpha_t)/np.sqrt(1-alpha_bar_t)) * epsilon_pred) + z * sigma_t
            if (t-1) % save_every_n_steps == 0:
                saved.append((t-1, xt.clone()))
        return xt, saved
