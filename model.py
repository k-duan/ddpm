import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, max_t: int, pos_emb: bool = True, n_channels: int = 3):
        super().__init__()
        self._conv1 =  nn.Conv2d(n_channels, 64, 3, 1, padding='same')
        self._conv1a = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._conv1b = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._conv1c = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._conv1d = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._ln1 = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])
        self._ln1a = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])
        self._ln1b = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])
        self._ln1c = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])
        self._ln1d = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])
        self._te1 = nn.Embedding(max_t, 64 * 16 * 16)
        self._pos_emb = pos_emb

        self._conv2 = nn.Conv2d(64, 128, 3, 1, padding='same')
        self._conv2a = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._conv2b = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._conv2c = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._conv2d = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._ln2 = nn.GroupNorm(32, 128)  # nn.LayerNorm([128, 16, 16])
        self._ln2a = nn.GroupNorm(32, 128)   # nn.LayerNorm([128, 16, 16])
        self._ln2b = nn.GroupNorm(32, 128)  # nn.LayerNorm([128, 16, 16])
        self._ln2c = nn.GroupNorm(32, 128)  # nn.LayerNorm([128, 16, 16])
        self._ln2d = nn.GroupNorm(32, 128)  # nn.LayerNorm([128, 16, 16])

        self._conv3 = nn.Conv2d(128, 256, 3, 1, padding='same')
        self._conv3a = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._conv3b = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._conv3c = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._conv3d = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._ln3 = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])
        self._ln3a = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])
        self._ln3b = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])
        self._ln3c = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])
        self._ln3d = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])

        self._convb = nn.Conv2d(256, 512, 3, 1, padding='same')
        self._convb1 = nn.Conv2d(512, 512, 3, 1, padding='same')
        self._convc1 = nn.Conv2d(512, 512, 3, 1, padding='same')
        self._convd1 = nn.Conv2d(512, 512, 3, 1, padding='same')
        self._lnb = nn.GroupNorm(32, 512)  # nn.LayerNorm([512, 4, 4])
        self._lnb1 = nn.GroupNorm(32, 512)  # nn.LayerNorm([512, 4, 4])
        self._lnc1 = nn.GroupNorm(32, 512)  # nn.LayerNorm([512, 4, 4])
        self._lnd1 = nn.GroupNorm(32, 512)  # nn.LayerNorm([512, 4, 4])

        self._deconv1 = nn.ConvTranspose2d(512+256, 256, 2, 2)
        self._deconv1a = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._deconv1b = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._deconv1c = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._deconv1d = nn.Conv2d(256, 256, 3, 1, padding='same')
        self._ln4 = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])
        self._ln4a = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])
        self._ln4b = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])
        self._ln4c = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])
        self._ln4d = nn.GroupNorm(32, 256)  # nn.LayerNorm([256, 8, 8])

        self._deconv2 = nn.ConvTranspose2d(256+128, 128, 2, 2)
        self._deconv2a = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._deconv2b = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._deconv2c = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._deconv2d = nn.Conv2d(128, 128, 3, 1, padding='same')
        self._ln5 = nn.GroupNorm(32, 128)  # nn.LayerNorm([128, 16, 16])
        self._ln5a = nn.GroupNorm(32, 128)  # nn.LayerNorm([128, 16, 16])
        self._ln5b = nn.GroupNorm(32, 128)  # nn.LayerNorm([128, 16, 16])
        self._ln5c = nn.GroupNorm(32, 128)  # nn.LayerNorm([128, 16, 16])
        self._ln5d = nn.GroupNorm(32, 128)  # nn.LayerNorm([128, 16, 16])

        self._deconv3 = nn.ConvTranspose2d(128+64, 64, 2, 2)
        self._deconv3a = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._deconv3b = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._deconv3c = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._deconv3d = nn.Conv2d(64, 64, 3, 1, padding='same')
        self._ln6 = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])
        self._ln6a = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])
        self._ln6b = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])
        self._ln6c = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])
        self._ln6d = nn.GroupNorm(32, 64)  # nn.LayerNorm([64, 32, 32])

        self._final_conv = nn.Conv2d(64, n_channels, 3, 1, padding='same')

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x1 = self._conv1(x)
        x1 = self._ln1(x1)
        x1 = nn.functional.silu(x1)
        x1 = self._conv1a(x1)
        x1 = self._ln1a(x1)
        x1 = nn.functional.silu(x1)
        x1 = self._conv1b(x1)
        x1 = self._ln1b(x1)
        x1 = nn.functional.silu(x1)
        x1 = self._conv1c(x1)
        x1 = self._ln1c(x1)
        x1 = nn.functional.silu(x1)
        x1 = self._conv1d(x1)
        x1 = self._ln1d(x1)
        x1 = nn.functional.silu(x1)
        x1 = nn.functional.max_pool2d(x1, 2)  # Bx3x32x32 -> Bx64x16x16
        if self._pos_emb:
            x1 = x1 + self._te1(t).view(x.size(0), 64, 16, 16)

        x2 = self._conv2(x1)
        x2 = self._ln2(x2)
        x2 = nn.functional.silu(x2)
        x2 = self._conv2a(x2)
        x2 = self._ln2a(x2)
        x2 = nn.functional.silu(x2)
        x2 = self._conv2b(x2)
        x2 = self._ln2b(x2)
        x2 = nn.functional.silu(x2)
        x2 = self._conv2c(x2)
        x2 = self._ln2c(x2)
        x2 = nn.functional.silu(x2)
        x2 = self._conv2d(x2)
        x2 = self._ln2d(x2)
        x2 = nn.functional.silu(x2)
        x2 = nn.functional.max_pool2d(x2, 2)  # Bx64x16x16 -> Bx128x8x8

        x3 = self._conv3(x2)
        x3 = self._ln3(x3)
        x3 = nn.functional.silu(x3)
        x3 = self._conv3a(x3)
        x3 = self._ln3a(x3)
        x3 = nn.functional.silu(x3)
        x3 = self._conv3b(x3)
        x3 = self._ln3b(x3)
        x3 = nn.functional.silu(x3)
        x3 = self._conv3c(x3)
        x3 = self._ln3c(x3)
        x3 = nn.functional.silu(x3)
        x3 = self._conv3d(x3)
        x3 = self._ln3d(x3)
        x3 = nn.functional.silu(x3)
        x3 = nn.functional.max_pool2d(x3, 2)  # Bx128x8x8 -> Bx256x4x4

        x4 = self._convb(x3)
        x4 = self._lnb(x4)
        x4 = nn.functional.silu(x4)
        x4 = self._convb1(x4)
        x4 = self._lnb1(x4)
        x4 = nn.functional.silu(x4)
        x4 = self._convc1(x4)
        x4 = self._lnc1(x4)
        x4 = nn.functional.silu(x4)
        x4 = self._convd1(x4)
        x4 = self._lnd1(x4)
        x4 = nn.functional.silu(x4)

        y1 = self._deconv1(torch.cat([x4, x3], dim=1))  # Bx(512+256)x4x4 -> Bx256x8x8
        y1 = self._ln4(y1)
        y1 = nn.functional.silu(y1)
        y1 = self._deconv1a(y1)
        y1 = self._ln4a(y1)
        y1 = nn.functional.silu(y1)
        y1 = self._deconv1b(y1)
        y1 = self._ln4b(y1)
        y1 = nn.functional.silu(y1)
        y1 = self._deconv1c(y1)
        y1 = self._ln4c(y1)
        y1 = nn.functional.silu(y1)
        y1 = self._deconv1d(y1)
        y1 = self._ln4d(y1)
        y1 = nn.functional.silu(y1)

        y2 = self._deconv2(torch.cat([y1, x2], dim=1))  # Bx(256+128)x8x8 -> Bx128x16x16
        y2 = self._ln5(y2)
        y2 = nn.functional.silu(y2)
        y2 = self._deconv2a(y2)
        y2 = self._ln5a(y2)
        y2 = nn.functional.silu(y2)
        y2 = self._deconv2b(y2)
        y2 = self._ln5b(y2)
        y2 = nn.functional.silu(y2)
        y2 = self._deconv2c(y2)
        y2 = self._ln5c(y2)
        y2 = nn.functional.silu(y2)
        y2 = self._deconv2d(y2)
        y2 = self._ln5d(y2)
        y2 = nn.functional.silu(y2)

        y3 = self._deconv3(torch.cat([y2, x1], dim=1))  # Bx(128+64)x16x16 -> Bx64x32x32
        y3 = self._ln6(y3)
        y3 = nn.functional.silu(y3)
        y3 = self._deconv3a(y3)
        y3 = self._ln6a(y3)
        y3 = nn.functional.silu(y3)
        y3 = self._deconv3b(y3)
        y3 = self._ln6b(y3)
        y3 = nn.functional.silu(y3)
        y3 = self._deconv3c(y3)
        y3 = self._ln6c(y3)
        y3 = nn.functional.silu(y3)
        y3 = self._deconv3d(y3)
        y3 = self._ln6d(y3)
        y3 = nn.functional.silu(y3)

        y = self._final_conv(y3)
        y = torch.nn.functional.tanh(y)
        return y

class DDPM(nn.Module):
    def __init__(self, max_t: int = 1000, pos_emb: bool = True, n_channels: int = 3):
        super().__init__()
        self._unet = UNet(max_t, pos_emb, n_channels)
        self._max_t = max_t
        self._beta_schedule = torch.linspace(0.0001, 0.02, max_t)

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
        return self._beta_schedule[t]

    @torch.no_grad()
    def alpha_bar_t(self, t: torch.Tensor) -> torch.Tensor:
        prods = torch.ones_like(self._beta_schedule)
        for i in range(self._max_t):
            prods[i] = prods[i-1] * (1-self._beta_schedule[i]) if i > 0 else 1 - self._beta_schedule[0]
        return prods[t]

    @torch.no_grad()
    def sample(self, n: int = 16, n_channels: int = 3, save_every_n_steps: int = 100) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        xt = torch.randn(n, n_channels, 32, 32)
        saved = [(self._max_t, xt.clone())]
        for t in reversed(range(1, self._max_t)):
            ts = torch.full((n,), t)
            z = torch.randn_like(xt) if t > 1 else 0
            alpha_t = 1 - self.beta_t(ts)
            alpha_bar_t = self.alpha_bar_t(ts)
            epsilon_pred = self._unet(xt, ts)
            sigma_t = torch.sqrt(1-alpha_t)
            xt = (1/torch.sqrt(alpha_t)).view(n, 1, 1, 1) * (xt - ((1-alpha_t)/torch.sqrt(1-alpha_bar_t)).view(n, 1, 1, 1) * epsilon_pred) + z * sigma_t.view(n, 1, 1, 1)
            if (t-1) % save_every_n_steps == 0:
                saved.append((t-1, xt.clone()))
        return xt, saved
