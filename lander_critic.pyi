"""Type stubs for lander_critic — learned V/Q critic for KTO lunar lander.

Shared-trunk architecture with dual heads:
  - V(s): value of state s = (obs_t, obs_prev, t_remaining)
  - Q(s, a): value of state s under plan a (encoded spline control points + T)

Reward definition (replaces LunarLander-v3 default):
  - r = 0          for all non-terminal steps
  - r = -t_elapsed for terminal steps (successful landing)
  - r = -t_elapsed - 10  for terminal steps (crash / timeout / e-stop)

This means V* ≈ -(expected time to land) for feasible states,
and Q* ranks plans by expected total cost (faster landing = less negative).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn

from kto_lander import Plan


# ===========================================================================
# State and action encoding
# ===========================================================================

@dataclass
class CriticState:
    """Encodes the critic's state representation.

    obs_t:        current gym observation (8,)
    obs_prev:     previous-step gym observation (8,)
    t_remaining:  seconds until episode timeout (scalar)
    """
    obs_t: np.ndarray         # (8,)
    obs_prev: np.ndarray      # (8,)
    t_remaining: float

    def to_tensor(self) -> torch.Tensor:
        """Flatten to (17,) tensor: [obs_t(8), obs_prev(8), t_remaining(1)]."""
        ...

    @staticmethod
    def from_tensor(t: torch.Tensor) -> CriticState: ...


STATE_DIM: int   # 17


@dataclass
class PlanEncoding:
    """Encodes a spline Plan as a fixed-size action vector.

    control_points: flattened (x,y) B-spline control points (num_cp * 2,)
    duration:       plan duration T (scalar)
    """
    control_points: np.ndarray   # (num_control_points * 2,)
    duration: float

    def to_tensor(self) -> torch.Tensor:
        """Flatten to (num_control_points * 2 + 1,) tensor."""
        ...

    @staticmethod
    def from_plan(plan: Plan) -> PlanEncoding:
        """Extract control points and T from a Plan object."""
        ...


DEFAULT_NUM_CONTROL_POINTS: int  # 12
ACTION_DIM: int                  # 25 = 12*2 + 1


# ===========================================================================
# Reward
# ===========================================================================

class Outcome(Enum):
    IN_PROGRESS = 0
    LANDED = 1
    FAILED = 2    # crash, timeout, or e-stop


def reward(
    outcome: Outcome,
    t_elapsed: float,
) -> float:
    """Compute step reward.

    Returns 0 for non-terminal, -t_elapsed for success,
    -(t_elapsed + 10) for failure.
    """
    ...


# ===========================================================================
# Model
# ===========================================================================

class TrunkType(Enum):
    MLP = "mlp"
    TRANSFORMER = "transformer"


@dataclass
class CriticConfig:
    """Hyperparameters for the critic network.

    trunk_type:    "mlp" or "transformer"
    trunk_hidden:  hidden dim for shared trunk (default 128)
    trunk_layers:  number of trunk layers (default 4 for MLP, 3 for transformer)
    head_hidden:   hidden dim for V/Q heads (default 64)
    head_layers:   number of head layers (default 2)
    n_heads:       attention heads (transformer only, default 4)
    dropout:       dropout rate (default 0.0)

    Target param budget: ~500k total.
    """
    trunk_type: TrunkType = ...
    trunk_hidden: int = ...
    trunk_layers: int = ...
    head_hidden: int = ...
    head_layers: int = ...
    n_heads: int = ...
    dropout: float = ...

    @property
    def param_count_estimate(self) -> int:
        """Approximate total parameter count."""
        ...


class LanderCritic(nn.Module):
    """Shared-trunk critic with dual V and Q heads.

    Architecture:
        state ──► [state_encoder] ──► trunk_embedding
                                          │
                        ┌─────────────────┼──────────────────┐
                        ▼                                    ▼
                    V_head(trunk_embedding)         Q_head(trunk_embedding, action)
                        │                                    │
                        ▼                                    ▼
                    V(s) scalar                        Q(s,a) scalar

    The state encoder projects (17,) → trunk_hidden.
    For Q, the action encoding is concatenated with trunk output
    before entering the Q head.
    """
    config: CriticConfig

    def __init__(self, config: CriticConfig | None = None) -> None: ...

    def forward_v(self, state: torch.Tensor) -> torch.Tensor:
        """Compute V(s). state: (batch, 17) → (batch, 1)."""
        ...

    def forward_q(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q(s, a). state: (batch, 17), action: (batch, 25) → (batch, 1)."""
        ...

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """If action is None, return V(s); otherwise return Q(s, a)."""
        ...


# ===========================================================================
# Rollout database
# ===========================================================================

@dataclass
class Transition:
    """Single environment transition for training."""
    state: CriticState
    action: PlanEncoding | None      # None for V-only transitions
    reward: float                    # 0 non-terminal, -t or -(t+10) terminal
    next_state: CriticState | None   # None if terminal
    outcome: Outcome
    t_elapsed: float                 # wall time into episode
    episode_id: int
    step_idx: int


class RolloutDB:
    """On-disk rollout database backed by memory-mapped numpy arrays.

    Storage layout:
        db_path/
            meta.json          # episode count, schema version, config
            states.npy         # (N, 17) float32
            actions.npy        # (N, 25) float32  (NaN rows for V-only)
            rewards.npy        # (N,)   float32
            terminals.npy      # (N,)   bool
            episode_ids.npy    # (N,)   int32
            step_indices.npy   # (N,)   int32
    """

    def __init__(self, db_path: Path, readonly: bool = False) -> None: ...

    @staticmethod
    def create(db_path: Path, capacity: int = 100_000) -> RolloutDB:
        """Initialize a new empty database with pre-allocated arrays."""
        ...

    def append_episode(self, transitions: Sequence[Transition]) -> int:
        """Append a full episode. Returns episode_id."""
        ...

    def sample_batch(
        self,
        batch_size: int,
        terminal_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch. Returns dict with keys:
        'state', 'action', 'reward', 'next_state', 'terminal'.
        """
        ...

    def __len__(self) -> int: ...

    @property
    def num_episodes(self) -> int: ...

    def episode_returns(self) -> np.ndarray:
        """Return total return per episode (N_episodes,)."""
        ...


# ===========================================================================
# Training
# ===========================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    lr: float = ...               # default 3e-4
    batch_size: int = ...         # default 256
    gamma: float = ...            # default 1.0 (undiscounted — reward is already time)
    tau: float = ...              # target network soft-update rate, default 0.005
    v_coeff: float = ...          # loss weight for V head, default 1.0
    q_coeff: float = ...          # loss weight for Q head, default 1.0
    max_grad_norm: float = ...    # gradient clipping, default 1.0
    epochs: int = ...             # default 100
    steps_per_epoch: int = ...    # default 500
    val_frac: float = ...        # default 0.1
    checkpoint_dir: Path = ...


class Trainer:
    """Trains LanderCritic with TD(0) on V and Q heads jointly.

    V loss:  MSE( V(s), r + γ·V_target(s') )   — bootstrap from target net
    Q loss:  MSE( Q(s,a), r + γ·V_target(s') )  — Q bootstraps from V target

    Uses a slowly-updated target network for stability.
    """

    def __init__(
        self,
        critic: LanderCritic,
        db: RolloutDB,
        config: TrainConfig | None = None,
    ) -> None: ...

    def train(self, callback: TrainCallback | None = None) -> TrainResult: ...

    def save_checkpoint(self, path: Path) -> None: ...

    def load_checkpoint(self, path: Path) -> None: ...


@dataclass
class TrainResult:
    """Summary of a training run."""
    v_loss_final: float
    q_loss_final: float
    v_loss_history: list[float]
    q_loss_history: list[float]
    val_v_loss: float
    val_q_loss: float
    total_steps: int
    best_epoch: int


class TrainCallback:
    """Optional callback for training loop hooks."""
    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None: ...
    def on_batch_end(self, step: int, loss: float) -> None: ...


# ===========================================================================
# Inference helpers
# ===========================================================================

def rank_plans(
    critic: LanderCritic,
    state: CriticState,
    plans: Sequence[Plan],
) -> list[tuple[float, Plan]]:
    """Score candidate plans via Q(s, a) and return sorted (best first)."""
    ...


def is_state_feasible(
    critic: LanderCritic,
    state: CriticState,
    threshold: float = -15.0,
) -> bool:
    """Return True if V(s) > threshold (state is likely recoverable)."""
    ...
