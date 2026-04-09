"""DDPM linear-beta noising + deterministic DDIM sampler (eta = 0)."""

import torch

T_TRAIN: int   # 1000


def make_schedule(
    T: int = T_TRAIN,
    device: torch.device = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(betas, alphas, alpha_bar)`` for the linear-beta schedule
    ``betas = linspace(1e-4, 0.02, T)``."""


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward noising step.

    ``x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise``

    Args:
        x0       : clean target tensor of any leading shape, last dim arbitrary.
        t        : (B,) int64 timestep indices.
        alpha_bar: (T,) cumulative product of alphas.
    Returns:
        ``(x_t, noise)`` — both same shape as ``x0``.
    """


def ddim_sample(
    model,
    cond_cps: torch.Tensor,    # (B, 10, 2)
    n_steps: int = 50,
    T: int = T_TRAIN,
    device: torch.device = ...,
) -> torch.Tensor:
    """Deterministic DDIM sampler (eta = 0).

    Iterates from t=T-1 down to t=0 over ``n_steps`` linearly-spaced timesteps,
    starting from x ~ N(0, I). Returns the sampled target CPs of shape
    (B, 10, 2) in the model's normalised CP space.
    """
