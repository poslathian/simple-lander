"""64-dim diffusion transformer for control-point continuation.

Inputs (per training example):
    cond_cps  : (B, 10, 2)   — clamped 10-CP B-spline of trajectory window [0, 1.5s]
    noisy_tgt : (B, 10, 2)   — noised target CPs (window [0.33, 1.83s])
    t         : (B,)         — diffusion timestep in [0, T)

Output:
    eps_pred  : (B, 10, 2)   — predicted noise on the target CPs

Architecture:
    - Linear 2 -> 64 embedding for every CP token
    - Learned positional embedding for the 10 cond + 10 target slots
    - Learned segment embedding (cond vs target)
    - Sinusoidal time embedding -> MLP -> 64, broadcast-added to all tokens
    - N pre-LN transformer blocks (multi-head self-attention + MLP)
    - Final LayerNorm + linear head 64 -> 2 over the target tokens
"""

import math

import torch
from torch import nn

D_MODEL = 64
N_HEADS = 4
N_BLOCKS = 4
N_CP = 10


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal embedding for scalar t (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_blocks: int = N_BLOCKS,
        n_cp: int = N_CP,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_cp = n_cp

        self.cp_embed = nn.Linear(2, d_model)

        # 10 cond positions + 10 target positions
        self.pos_embed = nn.Parameter(torch.zeros(2 * n_cp, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # 0 = cond segment, 1 = target segment
        self.seg_embed = nn.Parameter(torch.zeros(2, d_model))
        nn.init.trunc_normal_(self.seg_embed, std=0.02)

        # Time-step MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_blocks)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(
        self,
        cond_cps: torch.Tensor,   # (B, 10, 2)
        noisy_tgt: torch.Tensor,  # (B, 10, 2)
        t: torch.Tensor,          # (B,)
    ) -> torch.Tensor:
        B = cond_cps.shape[0]

        cond_tok = self.cp_embed(cond_cps) + self.seg_embed[0]
        tgt_tok = self.cp_embed(noisy_tgt) + self.seg_embed[1]

        x = torch.cat([cond_tok, tgt_tok], dim=1)         # (B, 20, d)
        x = x + self.pos_embed[None, :, :]                # (1, 20, d) -> broadcast

        # Time embedding broadcast across tokens
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.d_model))  # (B, d)
        x = x + t_emb[:, None, :]

        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)

        # Only the target tokens carry the prediction.
        out = self.head(x[:, self.n_cp :, :])  # (B, 10, 2)
        return out
