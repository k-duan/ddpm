import torch
from torch import nn


class BlockC(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, n_groups: int = 16):
        super().__init__()
        self._conv = nn.Conv2d(in_dims, out_dims, 3, 1, padding='same')
        self._norm = nn.GroupNorm(n_groups, out_dims)

    def forward(self, x):
        x = self._conv(x)
        x = self._norm(x)
        return nn.functional.silu(x)

class ConvBlock(nn.Module):
    def __init__(self, n_channels: int, n_dims: int, n_layers: int = 4, max_pool: bool = True):
        super().__init__()
        self._blocks = nn.ModuleList([BlockC(n_dims if i > 0 else n_channels, n_dims) for i in range(n_layers)])
        self._max_pool = max_pool

    def forward(self, x):
        for block in self._blocks:
            x = block(x)
        if self._max_pool:
            x = nn.functional.max_pool2d(x, 2)
        return x

class BlockD(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, n_groups: int = 16):
        super().__init__()
        self._deconv = nn.ConvTranspose2d(in_dims, out_dims, 2, 2)
        self._norm = nn.GroupNorm(n_groups, out_dims)

    def forward(self, x):
        x = self._deconv(x)
        x = self._norm(x)
        return nn.functional.silu(x)

class DeconvBlock(nn.Module):
    def __init__(self, n_channels: int, n_dims: int, n_layers: int = 4):
        super().__init__()
        self._blocks = nn.ModuleList([BlockC(n_dims, n_dims) if i > 0 else BlockD(n_channels, n_dims) for i in range(n_layers)])

    def forward(self, x):
        for block in self._blocks:
            x = block(x)
        return x

class UNetV2(nn.Module):
    def __init__(self, n_channels: int = 3):
        super().__init__()
        conv_dims = [(n_channels,64,True), (64,128,True), (128,256,True), (256,512,False)]
        deconv_dims = [(512+256,256), (256+128,128), (128+64,64)]
        self._conv_blocks = nn.ModuleList([ConvBlock(in_dim, out_dim, 4, max_pool) for in_dim, out_dim, max_pool in conv_dims])
        self._deconv_blocks = nn.ModuleList([DeconvBlock(in_dim, out_dim, 4) for in_dim, out_dim in deconv_dims])
        self._final_conv = nn.Conv2d(64, n_channels, 3, 1, padding='same')

    def forward(self, x, t: torch.Tensor):
        xi = [x]
        for block in self._conv_blocks:
            xi.append(block(xi[-1]))
        y = xi[-1]
        for i, block in enumerate(self._deconv_blocks):
            y = block(torch.cat([y, xi[len(xi)-i-2]], dim=1))
        y = self._final_conv(y)
        return torch.nn.functional.tanh(y)

class DDPM(nn.Module):
    def __init__(self, max_t: int = 1000, pos_emb: bool = True, n_channels: int = 3):
        super().__init__()
        self._unet = UNetV2(n_channels)
        self._max_t = max_t
        self._beta_schedule = torch.linspace(0.0001, 0.02, max_t, dtype=torch.float64)

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
        xt = torch.sqrt(alpha_bar_t).view(bs, 1, 1, 1) * x0 + torch.sqrt(1-alpha_bar_t).view(bs, 1, 1, 1) * epsilon
        return torch.clamp(xt, -1, 1)

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
        return self._beta_schedule[t].to(torch.float32)

    @torch.no_grad()
    def alpha_bar_t(self, t: torch.Tensor) -> torch.Tensor:
        prods = torch.ones_like(self._beta_schedule, dtype=torch.float64)
        for i in range(self._max_t):
            prods[i] = prods[i-1] * (1-self._beta_schedule[i]) if i > 0 else 1 - self._beta_schedule[0]
        return prods[t].to(torch.float32)

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
