"""Window-pair dataset built from a cached KTO position-trajectory pool.

Each cached plan is a 2-D (x, y) trajectory sampled at ``DT``.
For each plan we yield ``(cond_window, target_window)`` pairs where
``target`` is ``cond`` shifted forward by ``SHIFT_S`` seconds.
Both windows are fit with a clamped 10-CP cubic B-spline (see
``bspline.fit_window``) before being passed to the model.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

DT: float          # 0.02 s
WINDOW_S: float    # 1.5 s
SHIFT_S: float     # 0.33 s
WINDOW_LEN: int    # 76 samples (= round(WINDOW_S / DT) + 1)
SHIFT_STEPS: int   # 17 samples (= round(SHIFT_S / DT))


def load_pool(path: str) -> dict:
    """Unpickle the cached KTO pool. Each value is
    ``{"plan": {"x": np.ndarray, "y": np.ndarray, ...}, "n_steps": int}``.
    """


def extract_plans(pool: dict) -> list[np.ndarray]:
    """Stack each plan's (x, y) into a (T_i, 2) float32 array.
    Plans have variable length."""


def build_pairs(
    plans: list[np.ndarray],
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """For every plan, fit a 10-CP B-spline to every valid 1.5 s window and
    its 0.33 s-shifted future window.

    Args:
        plans: list of (T_i, 2) trajectories.
        stride: step in samples between successive window starts.
    Returns:
        cond_cps : (N_pairs, 10, 2) float32
        target_cps: (N_pairs, 10, 2) float32
    """


class WindowPairDataset(Dataset):
    """Translation-anchored, z-scored ``(cond_cps, target_cps)`` pairs.

    Each example is anchored by subtracting the cond window's first control
    point from BOTH windows so the model learns translation-invariant
    continuations. After anchoring, a single per-channel mean / std (computed
    over all anchored CPs) is applied. Use :meth:`denorm` to undo the z-score
    on a model output (the anchor cannot be undone — it must be re-added by the
    caller using the original cond CP[0]).
    """

    cond: np.ndarray   # (N, 10, 2) anchored & z-scored
    tgt: np.ndarray    # (N, 10, 2) anchored & z-scored
    mean: np.ndarray   # (2,) per-channel
    std: np.ndarray    # (2,) per-channel

    def __init__(self, cond_cps: np.ndarray, tgt_cps: np.ndarray) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(cond, target)`` tensors of shape (10, 2) each."""

    def denorm(self, cps_t: torch.Tensor) -> torch.Tensor:
        """Reverse the z-score normalisation on a tensor of CPs."""
