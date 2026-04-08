"""DiffusionMLP for position B-spline planning.

x_dim=30 (10 CPs x 3: x, y, theta), cond_dim=21 (20 state + 1 CFG).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

# ── Conditioning vector layout ────────────────────────────────────────────
STATE_DIM = 20
CFG_DIM = 1
COND_DIM = STATE_DIM + CFG_DIM  # 21
X_DIM = 30  # 10 CPs x 3
N_CPS = 10
N_CHANNELS = 3

CFG_START = STATE_DIM
CFG_END = COND_DIM


# ── Model components ─────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DiffusionMLP(nn.Module):
    """MLP denoiser for position B-spline diffusion."""

    def __init__(
        self,
        x_dim: int = X_DIM,
        cond_dim: int = COND_DIM,
        hidden: int = 256,
        n_blocks: int = 6,
        t_embed_dim: int = 64,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(t_embed_dim),
            nn.Linear(t_embed_dim, hidden),
            nn.SiLU(),
        )
        self.input_proj = nn.Linear(x_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden, x_dim)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        h = self.input_proj(x) + self.cond_proj(cond) + self.time_embed(t)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


# ── Cosine noise schedule ────────────────────────────────────────────────

class CosineSchedule:
    def __init__(self, T: int = 100, s: float = 0.008):
        self.T = T
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        self.alpha_bar = np.clip(alpha_bar, 1e-5, 1.0).astype(np.float32)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]


# ── DDIM sampler ────────────────────────────────────────────────────────

class DDIMSampler:
    def __init__(self, model: DiffusionMLP, schedule: CosineSchedule,
                 n_steps: int = 10, eta: float = 0.0):
        self.model = model
        self.schedule = schedule
        self.n_steps = n_steps
        self.eta = eta

        T = schedule.T
        self.timesteps = np.linspace(T, 0, n_steps + 1, dtype=int)[:-1]
        self.timesteps[-1] = max(self.timesteps[-1], 1)

    def _ddim_step(self, x, eps_pred, t_cur, t_prev, device):
        alpha_bar_t = torch.tensor(self.schedule.get_alpha_bar(t_cur), device=device)
        alpha_bar_prev = torch.tensor(self.schedule.get_alpha_bar(t_prev), device=device)

        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        if t_prev == 0:
            return x0_pred

        sigma = self.eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
        )
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_pred
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
        if self.eta > 0:
            x = x + sigma * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        device: str = "cpu",
    ) -> torch.Tensor:
        """DDIM sample conditioned on cond (includes outcome as input dim)."""
        B = cond.shape[0]
        x_dim = self.model.output_proj.out_features
        x = torch.randn(B, x_dim, device=device)

        for i in range(len(self.timesteps)):
            t_cur = self.timesteps[i]
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else 0
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps_pred = self.model(x, cond, t_batch)
            x = self._ddim_step(x, eps_pred, t_cur, t_prev, device)

        return x

    @torch.no_grad()
    def sample_cfg(
        self,
        cond: torch.Tensor,
        guidance_scale: float = 2.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Sample with classifier-free guidance (legacy, prefer sample())."""
        B = cond.shape[0]
        x_dim = self.model.output_proj.out_features
        x = torch.randn(B, x_dim, device=device)

        cond_uncond = cond.clone()
        cond_uncond[:, CFG_START:CFG_END] = 0.0

        for i in range(len(self.timesteps)):
            t_cur = self.timesteps[i]
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else 0
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

            eps_cond = self.model(x, cond, t_batch)
            eps_uncond = self.model(x, cond_uncond, t_batch)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            x = self._ddim_step(x, eps_pred, t_cur, t_prev, device)

        return x
