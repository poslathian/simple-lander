"""Build (input_window_cps, future_window_cps) training pairs from the cached
KTO position trajectories.

Each cached plan is a (251,) sequence of (x, y) at dt = 0.02 s, total 5.0 s.

For every plan we sample valid window starts t0 such that
[t0, t0 + 1.5 s] (the input/conditioning window) and
[t0 + 0.33, t0 + 0.33 + 1.5 s] (the predicted future window)
both fit inside the trajectory.
"""

import pickle

import numpy as np
import torch

from bspline import N_CP, fit_window

DT = 0.02
WINDOW_S = 1.5
SHIFT_S = 0.33
WINDOW_LEN = int(round(WINDOW_S / DT)) + 1   # 76 samples (inclusive endpoints)
SHIFT_STEPS = int(round(SHIFT_S / DT))       # 17 samples


def load_pool(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_plans(pool: dict) -> list[np.ndarray]:
    """Return list of (T_i, 2) arrays — plans have variable length."""
    return [
        np.stack([v["plan"]["x"], v["plan"]["y"]], axis=1).astype(np.float32)
        for v in pool.values()
    ]


def build_pairs(
    plans: list[np.ndarray],
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """For every plan, sample window pairs at every valid start.

    Returns:
        cond_cps : (N_pairs, 10, 2)
        target_cps: (N_pairs, 10, 2)
    """
    cond_list = []
    tgt_list = []
    for plan in plans:
        T = plan.shape[0]
        last_start = T - (WINDOW_LEN + SHIFT_STEPS)
        if last_start < 0:
            continue  # plan too short
        for s in range(0, last_start + 1, stride):
            cond_window = plan[s : s + WINDOW_LEN]
            tgt_window = plan[s + SHIFT_STEPS : s + SHIFT_STEPS + WINDOW_LEN]
            cond_list.append(fit_window(cond_window))
            tgt_list.append(fit_window(tgt_window))
    return (
        np.asarray(cond_list, dtype=np.float32),
        np.asarray(tgt_list, dtype=np.float32),
    )


class WindowPairDataset(torch.utils.data.Dataset):
    """Returns (cond_cps, target_cps) tensors of shape (10, 2) each, normalised
    by per-window centering — we subtract the cond window's first CP from BOTH
    cond and target so the model learns translation-invariant continuations.
    """

    def __init__(self, cond_cps: np.ndarray, tgt_cps: np.ndarray):
        # Translation-normalise: anchor at the cond window's CP[0].
        anchor = cond_cps[:, 0:1, :]  # (N, 1, 2)
        self.cond = (cond_cps - anchor).astype(np.float32)
        self.tgt = (tgt_cps - anchor).astype(np.float32)

        # Per-channel scale (computed once over all CPs).
        flat = np.concatenate(
            [self.cond.reshape(-1, 2), self.tgt.reshape(-1, 2)], axis=0
        )
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-6
        self.cond = (self.cond - self.mean) / self.std
        self.tgt = (self.tgt - self.mean) / self.std

    def __len__(self):
        return self.cond.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.cond[i]),  # (10, 2)
            torch.from_numpy(self.tgt[i]),   # (10, 2)
        )

    def denorm(self, cps_t: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=cps_t.dtype, device=cps_t.device)
        std = torch.as_tensor(self.std, dtype=cps_t.dtype, device=cps_t.device)
        return cps_t * std + mean
