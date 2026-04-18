"""Type stubs for diffusion_policy — DDPM/DDIM advantage-conditioned plan generator.

Replaces the Drake KTO solver with a 10-step DDIM denoising pass (~200k param MLP).
Trained via DDPM noise prediction on KTO demonstrations, then online improvement
with binarized advantage conditioning, HER, and structured replay sampling.

Training (DDPM):
  - Sample diffusion timestep t ~ U{0,...,T-1}
  - Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
  - Predict noise: loss = MSE(model(x_t, t, cond), eps)

Inference (10-step DDIM):
  - Start from x_T ~ N(0, I)
  - Iteratively denoise: x_{t-1} = f(x_t, predicted_noise, t)
  - 10 evenly-spaced steps from T to 0

Conditioning (18 dims):
  obs_t(8) + obs_prev(8) + t_remaining(1) + advantage(1)

Output (25 dims):
  12 B-spline control points x 2 (x,y) + duration T
  Same shape as PlanEncoding / KTO output.

Advantage is binarized sign(Q(s,a) - V(s)) in {-1, 0, +1}.
20% advantage dropout during training (mask to 0) for classifier-free guidance.
HER relabels advantage conditioning to match actual achieved advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from kto_lander import LanderParams, Plan
from lander_critic import (
    CriticState,
    LanderCritic,
    Outcome,
    PlanEncoding,
    RolloutDB,
    Transition,
)


# ===========================================================================
# Constants
# ===========================================================================

COND_DIM: int       # 18: obs_t(8) + obs_prev(8) + t_remaining(1) + advantage(1)
ACTION_DIM: int     # 25: 12*2 + 1
TIMESTEP_EMB_DIM: int  # 32: sinusoidal embedding of diffusion timestep


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class DPConfig:
    """Hyperparameters for the diffusion policy network.

    hidden_dim:         MLP hidden layer width (default 256)
    num_layers:         number of residual hidden blocks (default 4)
    dropout:            dropout rate in hidden layers (default 0.1)
    advantage_dropout:  rate of masking advantage to 0 during training (default 0.2)
    n_diffusion_steps:  total DDPM timesteps T for training (default 100)
    n_ddim_steps:       DDIM inference steps (default 10)
    beta_start:         linear noise schedule start (default 1e-4)
    beta_end:           linear noise schedule end (default 0.02)

    Target param budget: ~200k total.
    """
    hidden_dim: int = ...
    num_layers: int = ...
    dropout: float = ...
    advantage_dropout: float = ...
    n_diffusion_steps: int = ...
    n_ddim_steps: int = ...
    beta_start: float = ...
    beta_end: float = ...

    @property
    def param_count_estimate(self) -> int:
        """Approximate total parameter count."""
        ...


# ===========================================================================
# Conditioning
# ===========================================================================

@dataclass
class DPCondition:
    """Conditioning input for the diffusion policy.

    obs_t:        current gym observation (8,)
    obs_prev:     previous-step gym observation (8,)
    t_remaining:  seconds until episode timeout (scalar)
    advantage:    binarized advantage: -1 (bad), 0 (masked/unconditional), +1 (good)
    """
    obs_t: np.ndarray         # (8,)
    obs_prev: np.ndarray      # (8,)
    t_remaining: float
    advantage: float          # -1, 0, or +1

    def to_tensor(self) -> torch.Tensor:
        """Flatten to (18,) tensor: [obs_t(8), obs_prev(8), t_remaining(1), advantage(1)]."""
        ...

    @staticmethod
    def from_critic_state(cs: CriticState, advantage: float) -> DPCondition:
        """Construct from a CriticState + advantage scalar."""
        ...


# ===========================================================================
# Normalization
# ===========================================================================

@dataclass
class PlanNormalization:
    """Per-dimension mean/std for normalizing plan targets.

    Computed from KTO dataset. Network operates in normalized space;
    denormalize after DDIM sampling before plan reconstruction.
    """
    mean: np.ndarray    # (25,)
    std: np.ndarray     # (25,)

    def normalize(self, plans: torch.Tensor) -> torch.Tensor:
        """(batch, 25) -> normalized (batch, 25)."""
        ...

    def denormalize(self, plans: torch.Tensor) -> torch.Tensor:
        """(batch, 25) -> original scale (batch, 25)."""
        ...

    @staticmethod
    def fit(db: RolloutDB) -> PlanNormalization:
        """Compute mean/std from all valid (non-NaN) actions in the DB."""
        ...

    def save(self, path: Path) -> None: ...

    @staticmethod
    def load(path: Path) -> PlanNormalization: ...


# ===========================================================================
# Noise schedule
# ===========================================================================

class NoiseSchedule:
    """Linear beta schedule for DDPM.

    Precomputes alpha, alpha_bar, and sqrt terms for training and DDIM sampling.
    """
    n_steps: int
    betas: torch.Tensor           # (T,)
    alphas: torch.Tensor          # (T,)
    alpha_bars: torch.Tensor      # (T,)  cumulative product of alphas

    def __init__(self, n_steps: int, beta_start: float, beta_end: float) -> None: ...

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise."""
        ...

    def ddim_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> torch.Tensor:
        """One DDIM denoising step: x_{t_prev} from x_t and predicted noise."""
        ...


# ===========================================================================
# Model
# ===========================================================================

class DiffusionPolicy(nn.Module):
    """Noise-prediction MLP for DDPM training / DDIM inference.

    Architecture:
        [noisy_action(25) | timestep_emb(32) | cond(18)] = 75 dims
        -> Linear(75, 256) -> [LayerNorm -> SiLU -> Linear(256, 256) + residual] x 4
        -> Linear(256, 25)  (predicted noise)

    Timestep embedding: sinusoidal positional encoding projected through
    a learned linear layer.
    """
    config: DPConfig
    schedule: NoiseSchedule

    def __init__(self, config: DPConfig | None = None) -> None: ...

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise.

        x_t:  (batch, 25) noisy action
        t:    (batch,) integer diffusion timesteps
        cond: (batch, 18) conditioning
        Returns: (batch, 25) predicted noise epsilon
        """
        ...

    @torch.no_grad()
    def sample_ddim(
        self,
        cond: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """10-step DDIM sampling.

        cond: (batch, 18) conditioning
        Returns: (batch, 25) denoised plan in normalized space
        """
        ...

    @torch.no_grad()
    def generate_plan(
        self,
        condition: DPCondition,
        norm: PlanNormalization,
    ) -> PlanEncoding:
        """Single-sample inference: DDIM sample -> denormalize -> PlanEncoding."""
        ...


# ===========================================================================
# Plan reconstruction
# ===========================================================================

def encoding_to_plan(enc: PlanEncoding, gravity: float = ...) -> Plan:
    """Reconstruct a Drake BsplineTrajectory + Plan from a PlanEncoding.

    1. Reshape 24 control point values -> (12, 2) control points
    2. Build clamped uniform knot vector [0,0,0,0, T/9, 2T/9, .., 8T/9, T,T,T,T]
    3. BsplineBasis(order=4, knots) + BsplineTrajectory(basis, cps)
    4. Return Plan(traj, T, gravity)

    Compatible with kto_lander.control() for tracking.
    """
    ...


# ===========================================================================
# Extended RolloutDB
# ===========================================================================

class DPRolloutDB(RolloutDB):
    """Extends RolloutDB with per-episode metadata for structured sampling.

    Additional on-disk arrays:
        episode_advantages.npy      (max_episodes,) float32 -- Q(s0,a)-V(s0) per episode
        episode_outcomes.npy        (max_episodes,) int8    -- 0=failed, 1=landed
        episode_tracking_errors.npy (max_episodes,) float32 -- max tracking error
        episode_sources.npy         (max_episodes,) int8    -- 0=KTO, 1=DP
        episode_iterations.npy      (max_episodes,) int32   -- training iteration index
    """

    def __init__(self, db_path: Path, readonly: bool = False) -> None: ...

    @staticmethod
    def create(
        db_path: Path, capacity: int = 500_000, max_episodes: int = 50_000
    ) -> DPRolloutDB:
        """Initialize a new empty database with pre-allocated arrays."""
        ...

    @staticmethod
    def upgrade_from(existing: RolloutDB, new_path: Path) -> DPRolloutDB:
        """Copy an existing RolloutDB into a DPRolloutDB, adding metadata arrays.

        Sets source=KTO(0), advantage=0, iteration=0 for all existing episodes.
        """
        ...

    def append_episode_with_meta(
        self,
        transitions: Sequence[Transition],
        advantage: float,
        outcome: Outcome,
        max_tracking_error: float,
        source: int,           # 0=KTO, 1=DP
        iteration: int,
    ) -> int:
        """Append episode + per-episode metadata. Returns episode_id."""
        ...

    def sample_structured_batch(
        self,
        batch_size: int = 500,
        n_newest: int = 100,
        n_worst_recent: int = 100,
        n_edge_cases: int = 100,
        n_best_overall: int = 100,
        n_best_recent: int = 100,
        recent_window: int = 10,
        current_iteration: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Phase 3 structured sampling.

        1. Select episode IDs per bucket using metadata arrays
        2. Union and dedup episode IDs
        3. Sample transitions uniformly from selected episodes
        4. Return batch dict with keys: state, action, reward, mc_return,
           terminal, episode_advantage, episode_outcome
        """
        ...


# ===========================================================================
# Training
# ===========================================================================

@dataclass
class DPTrainConfig:
    """Training hyperparameters for phases 2 and 3."""
    # Phase
    phase: int = ...                   # 2 = behavioral cloning, 3 = online

    # Common
    lr: float = ...                    # default 3e-4
    batch_size: int = ...              # default 256
    epochs: int = ...                  # default 100
    steps_per_epoch: int = ...         # default 200
    max_grad_norm: float = ...         # default 1.0
    advantage_dropout: float = ...     # default 0.2

    # Phase 3
    episodes_per_iteration: int = ...  # default 500
    n_iterations: int = ...            # default 20

    # HER — relabel advantage conditioning to match actual achieved advantage
    her_relabel_prob: float = ...      # default 0.5

    # Evaluation
    holdout_seeds: list[int] = ...     # default range(5000, 5100)

    checkpoint_dir: Path = ...


@dataclass
class DPTrainResult:
    """Summary of a training run."""
    loss_history: list[float]
    val_loss: float
    landing_rate: float            # on holdout seeds
    mean_advantage: float          # E[Q(s, a_dp) - V(s)]
    mean_return: float             # E[episode return]
    planning_latency_ms: float     # mean wall-clock per plan
    best_epoch: int


class DPTrainer:
    """Trains DiffusionPolicy through phases 2 and 3.

    Phase 2 (behavioral cloning via DDPM):
      - Uniform sampling from DB (all KTO episodes)
      - DDPM noise prediction loss
      - Advantage conditioning active but signal is near-zero (all KTO)

    Phase 3 (online improvement):
      - Iterative: collect DP episodes via DDIM, update critic, update actor
      - Structured replay sampling with 5 buckets
      - HER: relabel advantage conditioning to match actual achieved advantage
      - Evaluate on holdout seeds each iteration
    """

    def __init__(
        self,
        policy: DiffusionPolicy,
        critic: LanderCritic,
        db: DPRolloutDB,
        norm: PlanNormalization,
        config: DPTrainConfig | None = None,
    ) -> None: ...

    def train_phase2(self) -> DPTrainResult:
        """Run phase 2: behavioral cloning of KTO plans via DDPM."""
        ...

    def train_phase3(self) -> DPTrainResult:
        """Run phase 3: online improvement loop."""
        ...

    def save_checkpoint(self, path: Path) -> None:
        """Save policy, optimizer, normalization, noise schedule, and config."""
        ...

    def load_checkpoint(self, path: Path) -> None: ...


# ===========================================================================
# Episode runner
# ===========================================================================

def run_dp_episode(
    policy: DiffusionPolicy,
    critic: LanderCritic,
    norm: PlanNormalization,
    render: bool = False,
    seed: int | None = None,
    verbose: bool = True,
    advantage_cond: float = 1.0,
) -> dict[str, float | int | bool]:
    """Run one episode using DP (DDIM) for planning + kto_lander.control() for tracking.

    Flow:
      1. warmup_and_snapshot()
      2. Build DPCondition from state + advantage_cond
      3. policy.generate_plan(condition, norm) -> PlanEncoding  (10-step DDIM)
      4. encoding_to_plan(enc) -> Plan
      5. Track with control(state, t, plan, params, gains)
      6. Return metrics dict (same keys as kto_lander.run_episode + planning_ms)
    """
    ...


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    """CLI with subcommands: phase2, phase3, evaluate, run, compare."""
    ...
