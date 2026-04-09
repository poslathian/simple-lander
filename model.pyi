"""64-dim diffusion transformer for B-spline control-point continuation.

Inputs (per training/inference call):
    cond_cps  : (B, 10, 2)   — clamped 10-CP B-spline of trajectory window [0, 1.5 s]
    noisy_tgt : (B, 10, 2)   — noised target CPs (window [0.33, 1.83 s])
    t         : (B,) int64   — diffusion timestep in [0, T)
Output:
    eps_pred  : (B, 10, 2)   — predicted noise on the target CPs

Architecture:
    - Linear 2 -> d_model embedding for every CP token
    - Learned positional embedding for the 10 cond + 10 target slots
    - Learned segment embedding (cond vs target)
    - Sinusoidal time embedding -> MLP -> d_model, broadcast-added to all tokens
    - n_blocks pre-LN transformer blocks (multi-head self-attention + MLP)
    - Final LayerNorm + linear head d_model -> 2 over the target tokens
"""

import torch
from torch import nn

D_MODEL: int    # 64
N_HEADS: int    # 4
N_BLOCKS: int   # 4
N_CP: int       # 10


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal embedding for a scalar timestep
    tensor of shape (B,) -> (B, dim)."""


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LN -> MHA -> residual -> LN -> MLP -> residual."""

    def __init__(self, d_model: int, n_heads: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class DiffusionTransformer(nn.Module):
    """64-dim diffusion transformer that predicts the eps noise applied to the
    10 target control points conditioned on the 10 input (cond) control points.
    """

    d_model: int
    n_cp: int

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_blocks: int = N_BLOCKS,
        n_cp: int = N_CP,
    ) -> None: ...

    def forward(
        self,
        cond_cps: torch.Tensor,    # (B, 10, 2)
        noisy_tgt: torch.Tensor,   # (B, 10, 2)
        t: torch.Tensor,           # (B,)
    ) -> torch.Tensor:
        """Returns predicted noise of shape (B, 10, 2) on the target CPs."""
