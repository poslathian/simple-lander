"""DiffusionMLP with CosineSchedule, DDIMSampler, and CFG for thrust B-spline planning.

Adapted from lunar-remote's diffusion/thrust/model.py.
x_dim=20 (10 B-spline CPs x 2), cond_dim=131, action_horizon is an input (not predicted).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# World constants (from _private/lunar_lander.py)
# ---------------------------------------------------------------------------
W = 30.0
H = 20.0
PAD_CX = 15.0
PAD_Y = 5.0
OBS_RADIUS = 0.75
DT = 0.02
OBS_X_NORM_OFFSET = 15.0
OBS_X_NORM_SCALE = 15.0
OBS_Y_NORM_OFFSET = 5.6
OBS_Y_NORM_SCALE = 10.0

# Conditioning vector layout
STATE_DIM = 119
CFG_DIM = 12
COND_DIM = STATE_DIM + CFG_DIM  # 131
X_DIM = 30  # 15 CPs x 2

# CFG conditioning indices (last 12 dims of cond vector)
CFG_START = STATE_DIM
CFG_END = COND_DIM


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

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
    """MLP denoiser for thrust B-spline diffusion.

    Predicts noise epsilon given noisy x, conditioning, and timestep.
    """

    def __init__(
        self,
        x_dim: int = X_DIM,
        cond_dim: int = COND_DIM,
        hidden: int = 512,
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


# ---------------------------------------------------------------------------
# Cosine noise schedule
# ---------------------------------------------------------------------------

class CosineSchedule:
    def __init__(self, T: int = 100, s: float = 0.008):
        self.T = T
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        self.alpha_bar = np.clip(alpha_bar, 1e-5, 1.0).astype(np.float32)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]


# ---------------------------------------------------------------------------
# DDIM sampler with CFG and guidance-action projection
# ---------------------------------------------------------------------------

class DDIMSampler:
    """Deterministic DDIM sampler with classifier-free guidance and
    optional action bounding-box projection."""

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
    def sample(self, cond: torch.Tensor, device: str = "cpu") -> torch.Tensor:
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
        action_boxes: torch.Tensor | None = None,
        action_horizon: float | None = None,
        norm_stats: dict | None = None,
    ) -> torch.Tensor:
        """Sample with classifier-free guidance.

        Args:
            cond: (B, 131) full conditioning vector.
            guidance_scale: CFG weight w.
            action_boxes: (B, n_boxes, 6) [thrust_v, thrust_v_margin, thrust_h,
                          thrust_h_margin, thrust_t, thrust_t_margin] for projection.
            action_horizon: spline duration (needed for projection).
            norm_stats: dict with x_mean, x_std for denorm/renorm during projection.
        """
        B = cond.shape[0]
        x_dim = self.model.output_proj.out_features
        x = torch.randn(B, x_dim, device=device)

        # Unconditional cond: zero the CFG dims (last 12)
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

            # Project CPs to satisfy action bounding boxes after each step
            if action_boxes is not None and norm_stats is not None and action_horizon is not None:
                x = _project_action_boxes(x, action_boxes, action_horizon, norm_stats, device)

        return x


def _project_action_boxes(
    x_norm: torch.Tensor,
    action_boxes: torch.Tensor,
    action_horizon: float,
    norm_stats: dict,
    device: str,
) -> torch.Tensor:
    """Project normalized CPs so the resulting spline satisfies action bounding boxes.

    This is a soft projection: for each box, evaluate the spline at thrust_t,
    clamp to the box, then adjust the nearest CP.
    """
    x_mean = torch.tensor(norm_stats["x_mean"], device=device)
    x_std = torch.tensor(norm_stats["x_std"], device=device)
    x_raw = x_norm * x_std + x_mean  # (B, 20)

    B = x_raw.shape[0]
    n_cps = 15

    for b in range(B):
        cps_v = x_raw[b, :n_cps].clone()  # vertical CPs
        cps_h = x_raw[b, n_cps:2*n_cps].clone()  # horizontal CPs

        for box in action_boxes[b]:
            tv, tv_m, th, th_m, tt, tt_m = box.tolist()
            if tt_m == 0 and tv_m == 0 and th_m == 0:
                continue  # masked

            # Find nearest CP index to thrust_t
            frac = tt / action_horizon if action_horizon > 0 else 0.5
            cp_idx = int(round(frac * (n_cps - 1)))
            cp_idx = max(0, min(n_cps - 1, cp_idx))

            # Clamp vertical CP
            if tv_m > 0:
                lo_v, hi_v = tv - tv_m, tv + tv_m
                cps_v[cp_idx] = cps_v[cp_idx].clamp(lo_v, hi_v)

            # Clamp horizontal CP
            if th_m > 0:
                lo_h, hi_h = th - th_m, th + th_m
                cps_h[cp_idx] = cps_h[cp_idx].clamp(lo_h, hi_h)

        x_raw[b, :n_cps] = cps_v
        x_raw[b, n_cps:2*n_cps] = cps_h

    return (x_raw - x_mean) / x_std

