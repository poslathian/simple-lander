"""DDPM/DDIM advantage-conditioned diffusion policy for lunar lander.

Replaces Drake KTO solver with a 10-step DDIM denoising pass (~200k param MLP).
Trained via DDPM noise prediction on KTO demonstrations, then online improvement
with binarized advantage conditioning, HER, and structured replay sampling.

Usage:
    python diffusion_policy.py phase2 --db dp_rollout_db --critic-ckpt checkpoints/best.pt
    python diffusion_policy.py phase3 --db dp_rollout_db --dp-ckpt dp_checkpoints/best.pt
    python diffusion_policy.py evaluate --dp-ckpt dp_checkpoints/best.pt
    python diffusion_policy.py run --dp-ckpt dp_checkpoints/best.pt --render
    python diffusion_policy.py compare --dp-ckpt dp_checkpoints/best.pt --episodes 100
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kto_lander import (
    DT,
    GRAVITY,
    Plan,
    TrackerGains,
    control,
    draw_plan_overlay,
    obs_to_state,
    warmup_and_snapshot,
)
from lander_critic import (
    ACTION_DIM,
    STATE_DIM,
    CriticState,
    LanderCritic,
    Outcome,
    PlanEncoding,
    RolloutDB,
    Trainer as CriticTrainer,
    TrainConfig as CriticTrainConfig,
    Transition,
    reward,
)


# ===========================================================================
# Constants
# ===========================================================================

COND_DIM = 18        # obs_t(8) + obs_prev(8) + t_remaining(1) + advantage(1)
TIMESTEP_EMB_DIM = 1  # scalar t/T normalized to [0,1]


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class DPConfig:
    hidden_dim: int = 192
    num_layers: int = 3
    dropout: float = 0.1
    advantage_dropout: float = 1.0  # unconditional BC
    n_diffusion_steps: int = 20
    n_ddim_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 0.1

    @property
    def param_count_estimate(self) -> int:
        input_dim = COND_DIM
        p = input_dim * self.hidden_dim + self.hidden_dim
        for _ in range(self.num_layers):
            p += self.hidden_dim * self.hidden_dim + self.hidden_dim  # linear
            p += 2 * self.hidden_dim  # layernorm
        p += self.hidden_dim * ACTION_DIM + ACTION_DIM
        return p


# ===========================================================================
# Conditioning
# ===========================================================================

@dataclass
class DPCondition:
    obs_t: np.ndarray         # (8,)
    obs_prev: np.ndarray      # (8,)
    t_remaining: float
    advantage: float          # -1, 0, or +1

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            np.concatenate([self.obs_t, self.obs_prev, [self.t_remaining, self.advantage]]),
            dtype=torch.float32,
        )

    @staticmethod
    def from_critic_state(cs: CriticState, advantage: float) -> DPCondition:
        return DPCondition(
            obs_t=cs.obs_t,
            obs_prev=cs.obs_prev,
            t_remaining=cs.t_remaining,
            advantage=advantage,
        )


# ===========================================================================
# Normalization
# ===========================================================================

@dataclass
class PlanNormalization:
    mean: np.ndarray    # (25,)
    std: np.ndarray     # (25,)

    def normalize(self, plans: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(self.mean, dtype=torch.float32, device=plans.device)
        s = torch.tensor(self.std, dtype=torch.float32, device=plans.device)
        return (plans - m) / s

    def denormalize(self, plans: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(self.mean, dtype=torch.float32, device=plans.device)
        s = torch.tensor(self.std, dtype=torch.float32, device=plans.device)
        return plans * s + m

    @staticmethod
    def fit(db: RolloutDB) -> PlanNormalization:
        actions = db._actions[:db._size]
        valid = ~np.isnan(actions[:, 0])
        valid_actions = actions[valid]
        mean = valid_actions.mean(axis=0)
        std = valid_actions.std(axis=0)
        std = np.maximum(std, 1e-6)
        return PlanNormalization(mean=mean, std=std)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load(path: Path) -> PlanNormalization:
        d = np.load(path)
        return PlanNormalization(mean=d["mean"], std=d["std"])


# ===========================================================================
# Noise schedule
# ===========================================================================

class NoiseSchedule:
    def __init__(self, n_steps: int, beta_start: float, beta_end: float):
        self.n_steps = n_steps
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        ab = self.alpha_bars.to(x_0.device)
        sqrt_ab = torch.sqrt(ab[t]).unsqueeze(-1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - ab[t]).unsqueeze(-1)
        return sqrt_ab * x_0 + sqrt_one_minus_ab * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> torch.Tensor:
        ab = self.alpha_bars.to(x_t.device)
        alpha_bar_t = ab[t]
        alpha_bar_prev = ab[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        # DDIM deterministic step
        x_prev = (
            torch.sqrt(alpha_bar_prev) * x_0_pred
            + torch.sqrt(1.0 - alpha_bar_prev) * predicted_noise
        )
        return x_prev


# ===========================================================================
# Model
# ===========================================================================

class _SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.linear(F.silu(self.ln(x))))


class DiffusionPolicy(nn.Module):
    """Direct conditional plan predictor (obs → plan).

    Despite the name, this is no longer a diffusion model — it directly
    predicts normalized plan encodings from observation conditioning.
    Exploration noise is added at inference time via `noise_scale`.
    """

    def __init__(self, config: DPConfig | None = None):
        super().__init__()
        self.config = config or DPConfig()
        cfg = self.config

        self.schedule = NoiseSchedule(cfg.n_diffusion_steps, cfg.beta_start, cfg.beta_end)

        # Input: cond(18) only — no noisy action or timestep
        self.input_proj = nn.Linear(COND_DIM, cfg.hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            _ResidualBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(cfg.hidden_dim, ACTION_DIM)

    def forward(
        self,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict normalized plan directly from conditioning."""
        h = self.input_proj(cond)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)

    # Legacy forward signature for compatibility with callers passing (x_t, t, cond)
    def forward_diffusion(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(cond)

        return x

    @torch.no_grad()
    def generate_plan(
        self,
        condition: DPCondition,
        norm: PlanNormalization,
        noise_scale: float = 0.0,
    ) -> PlanEncoding:
        self.eval()
        cond = condition.to_tensor().unsqueeze(0)
        x_norm = self.forward(cond)
        if noise_scale > 0:
            x_norm = x_norm + noise_scale * torch.randn_like(x_norm)
        x = norm.denormalize(x_norm).squeeze(0).cpu().numpy()
        return PlanEncoding(
            control_points=x[:24],
            duration=float(x[24]),
        )


# ===========================================================================
# Plan reconstruction
# ===========================================================================

def encoding_to_plan(enc: PlanEncoding, gravity: float = GRAVITY) -> Plan:
    from pydrake.math import BsplineBasis
    from pydrake.trajectories import BsplineTrajectory

    T = max(float(enc.duration), 0.5)
    num_cp = 12
    order = 4
    # Clamped uniform knot vector over [0, T]
    n_internal = num_cp - order  # 8
    knots = ([0.0] * order
             + [T * (i + 1) / (n_internal + 1) for i in range(n_internal)]
             + [T] * order)

    basis = BsplineBasis(order=order, knots=knots)
    cps_flat = enc.control_points
    control_points = [np.array([[cps_flat[2 * i]], [cps_flat[2 * i + 1]]]) for i in range(num_cp)]

    traj = BsplineTrajectory(basis, control_points)
    return Plan(traj, T, gravity)


# ===========================================================================
# Extended RolloutDB
# ===========================================================================

class DPRolloutDB(RolloutDB):
    def __init__(self, db_path: Path, readonly: bool = False):
        super().__init__(db_path, readonly)
        mode = "r" if readonly else "r+"
        self._ep_advantages = np.load(db_path / "episode_advantages.npy", mmap_mode=mode)
        self._ep_outcomes = np.load(db_path / "episode_outcomes.npy", mmap_mode=mode)
        self._ep_tracking_errors = np.load(db_path / "episode_tracking_errors.npy", mmap_mode=mode)
        self._ep_sources = np.load(db_path / "episode_sources.npy", mmap_mode=mode)
        self._ep_iterations = np.load(db_path / "episode_iterations.npy", mmap_mode=mode)
        self._max_episodes = len(self._ep_advantages)

    @staticmethod
    def create(db_path: Path, capacity: int = 500_000, max_episodes: int = 50_000) -> DPRolloutDB:
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        np.save(db_path / "states.npy", np.zeros((capacity, STATE_DIM), dtype=np.float32))
        np.save(db_path / "actions.npy", np.full((capacity, ACTION_DIM), np.nan, dtype=np.float32))
        np.save(db_path / "rewards.npy", np.zeros(capacity, dtype=np.float32))
        np.save(db_path / "mc_returns.npy", np.zeros(capacity, dtype=np.float32))
        np.save(db_path / "next_states.npy", np.zeros((capacity, STATE_DIM), dtype=np.float32))
        np.save(db_path / "terminals.npy", np.zeros(capacity, dtype=bool))
        np.save(db_path / "episode_ids.npy", np.full(capacity, -1, dtype=np.int32))
        np.save(db_path / "step_indices.npy", np.zeros(capacity, dtype=np.int32))
        # Episode metadata
        np.save(db_path / "episode_advantages.npy", np.zeros(max_episodes, dtype=np.float32))
        np.save(db_path / "episode_outcomes.npy", np.zeros(max_episodes, dtype=np.int8))
        np.save(db_path / "episode_tracking_errors.npy", np.zeros(max_episodes, dtype=np.float32))
        np.save(db_path / "episode_sources.npy", np.zeros(max_episodes, dtype=np.int8))
        np.save(db_path / "episode_iterations.npy", np.zeros(max_episodes, dtype=np.int32))
        meta = {
            "capacity": capacity,
            "size": 0,
            "num_episodes": 0,
            "max_episodes": max_episodes,
            "version": 2,
        }
        with open(db_path / "meta.json", "w") as f:
            json.dump(meta, f)
        return DPRolloutDB(db_path)

    @staticmethod
    def upgrade_from(existing: RolloutDB, new_path: Path) -> DPRolloutDB:
        new_path = Path(new_path)
        new_path.mkdir(parents=True, exist_ok=True)
        old = existing.db_path
        for name in ["states.npy", "actions.npy", "rewards.npy", "mc_returns.npy",
                      "next_states.npy", "terminals.npy", "episode_ids.npy", "step_indices.npy"]:
            src = old / name
            if src.exists():
                shutil.copy2(src, new_path / name)
        n_eps = existing.num_episodes
        max_episodes = max(50_000, n_eps * 2)
        np.save(new_path / "episode_advantages.npy", np.zeros(max_episodes, dtype=np.float32))
        np.save(new_path / "episode_outcomes.npy", np.zeros(max_episodes, dtype=np.int8))
        np.save(new_path / "episode_tracking_errors.npy", np.zeros(max_episodes, dtype=np.float32))
        np.save(new_path / "episode_sources.npy", np.zeros(max_episodes, dtype=np.int8))
        np.save(new_path / "episode_iterations.npy", np.zeros(max_episodes, dtype=np.int32))
        meta = {
            "capacity": existing._capacity,
            "size": existing._size,
            "num_episodes": n_eps,
            "max_episodes": max_episodes,
            "version": 2,
        }
        with open(new_path / "meta.json", "w") as f:
            json.dump(meta, f)
        return DPRolloutDB(new_path)

    def append_episode_with_meta(
        self,
        transitions: Sequence[Transition],
        advantage: float,
        outcome: Outcome,
        max_tracking_error: float,
        source: int,
        iteration: int,
    ) -> int:
        ep_id = self.append_episode(transitions)
        if ep_id >= self._max_episodes:
            raise RuntimeError("Episode metadata arrays full")
        self._ep_advantages[ep_id] = advantage
        self._ep_outcomes[ep_id] = 1 if outcome == Outcome.LANDED else 0
        self._ep_tracking_errors[ep_id] = max_tracking_error
        self._ep_sources[ep_id] = source
        self._ep_iterations[ep_id] = iteration
        return ep_id

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
        n_eps = self._num_episodes
        if n_eps == 0:
            return self.sample_batch(batch_size)

        all_ids = np.arange(n_eps)
        advantages = self._ep_advantages[:n_eps]
        iterations = self._ep_iterations[:n_eps]
        tracking_errs = self._ep_tracking_errors[:n_eps]
        outcomes = self._ep_outcomes[:n_eps]

        recent_mask = iterations >= max(0, current_iteration - recent_window)

        selected = set()

        # Newest
        newest_ids = all_ids[max(0, n_eps - n_newest):]
        selected.update(newest_ids.tolist())

        # Worst recent (lowest advantage in recent window)
        if recent_mask.any():
            recent_ids = all_ids[recent_mask]
            recent_adv = advantages[recent_mask]
            worst_idx = np.argsort(recent_adv)[:n_worst_recent]
            selected.update(recent_ids[worst_idx].tolist())

        # Edge cases (highest tracking error + failures)
        edge_score = tracking_errs + 5.0 * (1 - outcomes)
        edge_idx = np.argsort(edge_score)[-n_edge_cases:]
        selected.update(all_ids[edge_idx].tolist())

        # Best overall
        best_idx = np.argsort(advantages)[-n_best_overall:]
        selected.update(all_ids[best_idx].tolist())

        # Best recent
        if recent_mask.any():
            recent_ids = all_ids[recent_mask]
            recent_adv = advantages[recent_mask]
            best_recent_idx = np.argsort(recent_adv)[-n_best_recent:]
            selected.update(recent_ids[best_recent_idx].tolist())

        selected_eps = np.array(sorted(selected))

        # Sample initial-state transitions from selected episodes
        ep_ids_col = self._episode_ids[:self._size]
        step_idx_col = self._step_indices[:self._size]
        mask = np.isin(ep_ids_col, selected_eps) & (step_idx_col == 0)
        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            return self.sample_batch(batch_size)

        chosen = np.random.choice(
            candidate_indices,
            size=min(batch_size, len(candidate_indices)),
            replace=len(candidate_indices) < batch_size,
        )

        batch = {
            "state": torch.tensor(self._states[chosen], dtype=torch.float32),
            "action": torch.tensor(self._actions[chosen], dtype=torch.float32),
            "reward": torch.tensor(self._rewards[chosen], dtype=torch.float32),
            "terminal": torch.tensor(self._terminals[chosen], dtype=torch.bool),
        }
        if self._mc_returns is not None:
            batch["mc_return"] = torch.tensor(self._mc_returns[chosen], dtype=torch.float32)
        # Add per-episode metadata for each transition
        trans_ep_ids = self._episode_ids[chosen]
        batch["episode_advantage"] = torch.tensor(
            self._ep_advantages[trans_ep_ids], dtype=torch.float32
        )
        batch["episode_outcome"] = torch.tensor(
            self._ep_outcomes[trans_ep_ids], dtype=torch.float32
        )
        return batch

    def sample_initial_states(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample only episode-initial transitions (step_idx=0).

        The DP generates a plan once per episode at the initial state,
        so training should focus on these transitions, not mid-episode frames.
        """
        step_indices = self._step_indices[:self._size]
        ep_ids = self._episode_ids[:self._size]
        # Only KTO episodes (< 1000) — phase3 stubs have mode-collapsed plans
        initial_mask = (step_indices == 0) & (self._mc_returns[:self._size] != 0.0)
        initial_indices = np.where(initial_mask)[0]

        if len(initial_indices) == 0:
            return self.sample_batch(batch_size)

        chosen = np.random.choice(
            initial_indices,
            size=min(batch_size, len(initial_indices)),
            replace=len(initial_indices) < batch_size,
        )

        state_data = self._states[chosen].copy()
        state_data[:, 16] = 0.0  # zero t_remaining — not meaningful at episode start
        batch = {
            "state": torch.tensor(state_data, dtype=torch.float32),
            "action": torch.tensor(self._actions[chosen], dtype=torch.float32),
            "reward": torch.tensor(self._rewards[chosen], dtype=torch.float32),
            "terminal": torch.tensor(self._terminals[chosen], dtype=torch.bool),
        }
        if self._mc_returns is not None:
            batch["mc_return"] = torch.tensor(self._mc_returns[chosen], dtype=torch.float32)
        trans_ep_ids = self._episode_ids[chosen]
        batch["episode_advantage"] = torch.tensor(
            self._ep_advantages[trans_ep_ids], dtype=torch.float32
        )
        batch["episode_outcome"] = torch.tensor(
            self._ep_outcomes[trans_ep_ids], dtype=torch.float32
        )
        return batch


# ===========================================================================
# Training
# ===========================================================================

@dataclass
class DPTrainConfig:
    phase: int = 2
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    steps_per_epoch: int = 4  # ~1 pass over 1000 samples per epoch
    max_grad_norm: float = 1.0
    advantage_dropout: float = 0.1
    # Phase 3
    episodes_per_iteration: int = 500
    n_iterations: int = 20
    # Critic
    critic_tau: float = 0.005  # EMA rate for target network soft update
    # HER
    her_relabel_prob: float = 0.5
    # Evaluation
    holdout_seeds: list[int] = field(default_factory=lambda: list(range(5000, 5100)))
    checkpoint_dir: Path = field(default_factory=lambda: Path("dp_checkpoints"))


@dataclass
class DPTrainResult:
    loss_history: list[float]
    val_loss: float
    landing_rate: float
    mean_advantage: float
    mean_return: float
    planning_latency_ms: float
    best_epoch: int


class DPTrainer:
    def __init__(
        self,
        policy: DiffusionPolicy,
        critic: LanderCritic,
        db: DPRolloutDB,
        norm: PlanNormalization,
        config: DPTrainConfig | None = None,
    ):
        self.policy = policy
        self.critic = critic
        self.db = db
        self.norm = norm
        self.config = config or DPTrainConfig()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.lr)

        # Target network for critic stability (EMA of critic weights)
        import copy
        self.critic_target = copy.deepcopy(critic)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    def _soft_update_critic_target(self):
        """Polyak averaging: target = tau * critic + (1-tau) * target."""
        tau = self.config.critic_tau
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def train_phase2(self) -> DPTrainResult:
        cfg = self.config
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg_dp = self.policy.config

        loss_hist = []
        best_val = float("inf")
        best_epoch = 0

        print(f"[phase2] training on {self.db.num_episodes} initial-state transitions "
              f"({len(self.db)} total frames, {self.db.num_episodes} episodes)")

        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            for step in range(cfg.steps_per_epoch):
                batch = self.db.sample_initial_states(cfg.batch_size)
                loss = self._train_step_inner(batch, cfg_dp)
                epoch_loss += loss

            avg_loss = epoch_loss / cfg.steps_per_epoch
            loss_hist.append(avg_loss)

            # Validation
            val_batch = self.db.sample_initial_states(min(1024, self.db.num_episodes))
            val_loss = self._validate(val_batch, cfg_dp)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                self.save_checkpoint(cfg.checkpoint_dir / "best.pt")

            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{cfg.epochs}  loss={avg_loss:.6f}  val={val_loss:.6f}")

        # Save last checkpoint too
        self.save_checkpoint(cfg.checkpoint_dir / "last.pt")

        # Final evaluation
        metrics = self._evaluate_holdout()
        print(f"[phase2] done — best epoch {best_epoch}, landing_rate={metrics['landing_rate']:.1%}")

        return DPTrainResult(
            loss_history=loss_hist,
            val_loss=best_val,
            landing_rate=metrics["landing_rate"],
            mean_advantage=metrics["mean_advantage"],
            mean_return=metrics["mean_return"],
            planning_latency_ms=metrics["planning_latency_ms"],
            best_epoch=best_epoch,
        )

    def _train_step_inner(self, batch: dict[str, torch.Tensor], cfg_dp: DPConfig) -> float:
        """Core DDPM training step."""
        self.policy.train()
        cfg = self.config
        schedule = self.policy.schedule

        states = batch["state"]
        actions = batch["action"]
        has_action = ~torch.isnan(actions[:, 0])
        if not has_action.any():
            return 0.0

        states = states[has_action]
        actions = actions[has_action]
        bs = states.shape[0]

        actions_norm = self.norm.normalize(actions)

        # Binarized advantage: did this plan beat the critic's prediction?
        # advantage = sign(MC_return - V(s))
        # Positive = plan did better than expected, negative = worse.
        if "mc_return" in batch:
            mc = batch["mc_return"][has_action]
            self.critic.eval()
            with torch.no_grad():
                v = self.critic.forward_v(states).squeeze(-1)
            advantages = torch.sign(mc - v)
        elif "episode_advantage" in batch:
            advantages = torch.sign(batch["episode_advantage"][has_action])
        else:
            advantages = torch.zeros(bs)

        # Advantage dropout
        adv_mask = torch.rand(bs) < cfg.advantage_dropout
        advantages[adv_mask] = 0.0

        # Build conditioning
        cond = torch.cat([states, advantages.unsqueeze(-1)], dim=-1)

        # Direct prediction: predict normalized plan from conditioning
        pred_plan = self.policy(cond)
        loss = F.mse_loss(pred_plan, actions_norm)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def _validate(self, batch: dict[str, torch.Tensor], cfg_dp: DPConfig) -> float:
        """Compute validation loss."""
        self.policy.eval()

        states = batch["state"]
        actions = batch["action"]
        has_action = ~torch.isnan(actions[:, 0])
        if not has_action.any():
            return 0.0

        states = states[has_action]
        actions = actions[has_action]
        bs = states.shape[0]

        actions_norm = self.norm.normalize(actions)

        if "mc_return" in batch:
            mc = batch["mc_return"][has_action]
            self.critic.eval()
            with torch.no_grad():
                v = self.critic.forward_v(states).squeeze(-1)
            advantages = torch.sign(mc - v)
        elif "episode_advantage" in batch:
            advantages = torch.sign(batch["episode_advantage"][has_action])
        else:
            advantages = torch.zeros(bs)

        cond = torch.cat([states, advantages.unsqueeze(-1)], dim=-1)

        with torch.no_grad():
            pred_plan = self.policy(cond)
            loss = F.mse_loss(pred_plan, actions_norm)
        return loss.item()

    def train_phase3(self) -> DPTrainResult:
        cfg = self.config
        cfg_dp = self.policy.config
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        loss_hist = []
        best_landing = 0.0
        best_epoch = 0
        total_epoch = 0

        # Evaluate baseline before any phase 3 changes
        baseline = self._evaluate_holdout()
        best_landing = baseline["landing_rate"]
        print(f"[phase3] baseline: landing_rate={best_landing:.1%}, "
              f"mean_adv={baseline['mean_advantage']:.3f}")
        self.save_checkpoint(cfg.checkpoint_dir / "best.pt")

        print(f"[phase3] starting online improvement, {cfg.n_iterations} iterations")
        _last_print = time.perf_counter()

        def _status(msg: str) -> None:
            nonlocal _last_print
            _last_print = time.perf_counter()
            print(msg, flush=True)

        def _maybe_tick(phase: str, detail: str) -> None:
            nonlocal _last_print
            now = time.perf_counter()
            if now - _last_print >= 60.0:
                _last_print = now
                print(f"  [{phase}] {detail}", flush=True)

        for iteration in range(cfg.n_iterations):
            iter_start = time.perf_counter()
            _status(f"\n--- iteration {iteration+1}/{cfg.n_iterations} ---")

            # 1. Collect DP episodes
            stats = self._collect_dp_episodes(
                cfg.episodes_per_iteration, iteration
            )
            collect_rate = stats["landed"] / max(stats["landed"] + stats["failed"], 1)
            _status(f"  collected {cfg.episodes_per_iteration} eps: "
                    f"landed={stats['landed']}, failed={stats['failed']} "
                    f"({100*collect_rate:.0f}%)")

            # 2. Update critic on all data
            _status("  updating critic...")
            critic_cfg = CriticTrainConfig(epochs=30, steps_per_epoch=100)
            critic_trainer = CriticTrainer(self.critic, self.db, critic_cfg)
            critic_trainer.train()
            # Soft-update target network
            self._soft_update_critic_target()
            _status("  critic updated (target network EMA applied)")

            # Validate critic against ground truth
            self._evaluate_critic(n_seeds=20)

            # 3. Update DP — very conservative: 3 epochs, 1/10th LR
            _status("  updating policy (3 epochs, lr=3e-5)...")
            # Temporarily lower LR for fine-tuning
            for pg in self.optimizer.param_groups:
                pg["lr"] = cfg.lr * 0.1
            for epoch in range(3):
                epoch_loss = 0.0
                for step in range(cfg.steps_per_epoch):
                    batch = self.db.sample_structured_batch(
                        batch_size=cfg.batch_size,
                        current_iteration=iteration,
                    )
                    loss = self._train_step_phase3(batch, cfg_dp, iteration)
                    epoch_loss += loss
                    _maybe_tick("policy", f"epoch {epoch+1}/20 step {step+1}/{cfg.steps_per_epoch}")
                avg_loss = epoch_loss / cfg.steps_per_epoch
                loss_hist.append(avg_loss)
                total_epoch += 1
            # Restore LR
            for pg in self.optimizer.param_groups:
                pg["lr"] = cfg.lr
            _status(f"  policy updated (loss={avg_loss:.6f})")

            # 4. Evaluate
            _status("  evaluating holdout...")
            metrics = self._evaluate_holdout()
            print(f"  holdout: landing_rate={metrics['landing_rate']:.1%}, "
                  f"mean_adv={metrics['mean_advantage']:.3f}, "
                  f"mean_return={metrics['mean_return']:.2f}")

            if metrics["landing_rate"] > best_landing:
                best_landing = metrics["landing_rate"]
                best_epoch = total_epoch
                self.save_checkpoint(cfg.checkpoint_dir / "best.pt")
                print(f"  new best! saved checkpoint")
            else:
                # Revert to best checkpoint to prevent catastrophic forgetting
                print(f"  no improvement (best={best_landing:.1%}), reverting to best checkpoint")
                self.load_checkpoint(cfg.checkpoint_dir / "best.pt")

            # Early stopping
            if metrics["landing_rate"] > 0.95 and metrics["mean_advantage"] > 0:
                print(f"  converged at iteration {iteration+1}")
                break

        return DPTrainResult(
            loss_history=loss_hist,
            val_loss=loss_hist[-1] if loss_hist else 0.0,
            landing_rate=best_landing,
            mean_advantage=metrics["mean_advantage"],
            mean_return=metrics["mean_return"],
            planning_latency_ms=metrics["planning_latency_ms"],
            best_epoch=best_epoch,
        )

    def _train_step_phase3(
        self, batch: dict[str, torch.Tensor], cfg_dp: DPConfig, iteration: int
    ) -> float:
        """Phase 3 training step with HER."""
        self.policy.train()
        cfg = self.config
        schedule = self.policy.schedule

        states = batch["state"]
        actions = batch["action"]
        has_action = ~torch.isnan(actions[:, 0])
        if not has_action.any():
            return 0.0

        states = states[has_action]
        actions = actions[has_action]
        bs = states.shape[0]

        actions_norm = self.norm.normalize(actions)

        # Compute advantages from critic
        self.critic.eval()
        with torch.no_grad():
            v = self.critic.forward_v(states).squeeze(-1)
            q = self.critic.forward_q(states, actions).squeeze(-1)
            advantages = torch.sign(q - v)

        # HER: relabel advantage to match actual achieved advantage
        if "episode_advantage" in batch:
            ep_adv = batch["episode_advantage"][has_action]
            actual_sign = torch.sign(ep_adv)
            her_mask = torch.rand(bs) < cfg.her_relabel_prob
            advantages = torch.where(her_mask, actual_sign, advantages)

        # Advantage dropout
        adv_mask = torch.rand(bs) < cfg.advantage_dropout
        advantages[adv_mask] = 0.0

        cond = torch.cat([states, advantages.unsqueeze(-1)], dim=-1)

        pred_plan = self.policy(cond)
        loss = F.mse_loss(pred_plan, actions_norm)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def _collect_dp_episodes(
        self, n_episodes: int, iteration: int
    ) -> dict[str, int]:
        import gymnasium as gym

        landed = 0
        failed = 0

        t_start = time.perf_counter()
        for ep in range(n_episodes):
            seed = 10_000 + iteration * n_episodes + ep
            env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
            env.reset(seed=seed)

            try:
                obs_raw, state0, params = warmup_and_snapshot(env, n_steps=5)
            except RuntimeError:
                env.close()
                failed += 1
                continue

            # Generate plan with DP
            cs = CriticState(
                obs_t=obs_raw.copy(),
                obs_prev=obs_raw.copy(),
                t_remaining=0.0,
            )
            condition = DPCondition.from_critic_state(cs, advantage=1.0)
            plan_enc = self.policy.generate_plan(condition, self.norm)
            try:
                plan = encoding_to_plan(plan_enc)
            except Exception:
                env.close()
                failed += 1
                continue

            plan_T = plan.T
            total_duration = max(3.0, plan_T + 1.0)
            n_steps = int(round(total_duration / DT))
            gains = TrackerGains()

            # Compute advantage for this plan
            self.critic.eval()
            with torch.no_grad():
                s_tensor = cs.to_tensor().unsqueeze(0)
                a_tensor = plan_enc.to_tensor().unsqueeze(0)
                v_val = self.critic.forward_v(s_tensor).item()
                q_val = self.critic.forward_q(s_tensor, a_tensor).item()
            advantage = q_val - v_val

            transitions = []
            obs_prev = obs_raw.copy()
            obs_cur = obs_raw
            t = 0.0
            terminated = False
            episode_landed = False
            estop = False
            max_tracking_error = 0.0

            window_steps = max(1, int(round(0.1 / DT)))
            err_hist: list[float] = []

            for k in range(n_steps):
                obs_state = obs_to_state(obs_cur)
                t_remaining = total_duration - t

                critic_state = CriticState(
                    obs_t=obs_cur.copy(),
                    obs_prev=obs_prev.copy(),
                    t_remaining=t_remaining,
                )

                action_ctrl, _ = control(obs_state, t, plan, params, gains)
                obs_next, _, terminated, truncated, _ = env.step(action_ctrl)
                t += DT

                # E-stop check
                r = plan(t)
                ns = obs_to_state(obs_next)
                e_x = ns[0] - r[0]
                e_y = ns[1] - r[1]
                e_th = ns[4] - r[4]
                e_norm = math.sqrt(e_x**2 + e_y**2 + (2.0 * e_th)**2)
                max_tracking_error = max(max_tracking_error, e_norm)
                err_hist.append(e_norm)
                if len(err_hist) > window_steps:
                    err_hist.pop(0)
                if e_norm > 2.0 or (
                    len(err_hist) == window_steps and all(e > 1.0 for e in err_hist)
                ):
                    estop = True

                if obs_next[6] > 0.5 and obs_next[7] > 0.5:
                    episode_landed = True

                is_terminal = terminated or truncated or estop or episode_landed
                if is_terminal:
                    outcome = Outcome.LANDED if episode_landed else Outcome.FAILED
                else:
                    outcome = Outcome.IN_PROGRESS

                r_val = reward(outcome, t)
                next_cs = CriticState(
                    obs_t=obs_next.copy(),
                    obs_prev=obs_cur.copy(),
                    t_remaining=max(0.0, total_duration - t),
                )

                transitions.append(Transition(
                    state=critic_state,
                    action=plan_enc,
                    reward=r_val,
                    next_state=None if is_terminal else next_cs,
                    outcome=outcome,
                    t_elapsed=t,
                    episode_id=0,
                    step_idx=k,
                ))

                obs_prev = obs_cur.copy()
                obs_cur = obs_next

                if is_terminal:
                    break

            env.close()

            ep_outcome = Outcome.LANDED if episode_landed else Outcome.FAILED
            # Only store initial transition for DP episodes (we only train on step_idx=0)
            # This saves ~200x DB capacity vs storing every frame
            initial_only = [transitions[0]] if transitions else transitions
            self.db.append_episode_with_meta(
                initial_only,
                advantage=advantage,
                outcome=ep_outcome,
                max_tracking_error=max_tracking_error,
                source=1,  # DP
                iteration=iteration,
            )

            if episode_landed:
                landed += 1
            else:
                failed += 1

            elapsed = time.perf_counter() - t_start
            if (ep + 1) % 25 == 0 or elapsed - getattr(self, '_last_collect_print', 0) >= 60:
                self._last_collect_print = elapsed
                rate = (ep + 1) / max(elapsed, 0.1)
                eta = (n_episodes - ep - 1) / rate
                print(f"    collected {ep+1}/{n_episodes} (landed={landed}, failed={failed}) "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]", flush=True)

        return {"landed": landed, "failed": failed}

    def _evaluate_holdout(self) -> dict[str, float]:
        seeds = self.config.holdout_seeds
        landed = 0
        total_return = 0.0
        total_advantage = 0.0
        total_latency = 0.0
        valid = 0

        for seed in seeds:
            result = run_dp_episode(
                self.policy, self.critic, self.norm,
                render=False, seed=seed, verbose=False, advantage_cond=0.0,
            )
            if result.get("planning_failed"):
                continue
            valid += 1
            if result["landed"]:
                landed += 1
                total_return += -result["t_elapsed"]
            else:
                total_return += -(result["t_elapsed"] + 10.0)
            total_advantage += result.get("advantage", 0.0)
            total_latency += result.get("planning_ms", 0.0)

        n = max(valid, 1)
        return {
            "landing_rate": landed / n,
            "mean_return": total_return / n,
            "mean_advantage": total_advantage / n,
            "planning_latency_ms": total_latency / n,
        }

    def _evaluate_critic(self, n_seeds: int = 30) -> dict[str, float]:
        """Check critic V(s) predictions against actual MC episode returns."""
        import gymnasium as gym

        seeds = self.config.holdout_seeds[:n_seeds]
        v_preds = []
        actual_returns = []

        self.critic.eval()
        for seed in seeds:
            env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
            env.reset(seed=seed)
            try:
                obs_raw, state0, params = warmup_and_snapshot(env, n_steps=5)
            except RuntimeError:
                env.close()
                continue

            cs = CriticState(obs_t=obs_raw.copy(), obs_prev=obs_raw.copy(), t_remaining=10.0)
            with torch.no_grad():
                v = self.critic.forward_v(cs.to_tensor().unsqueeze(0)).item()
            v_preds.append(v)

            # Roll out with DP to get actual return (zero t_remaining for DP conditioning)
            cs_dp = CriticState(obs_t=obs_raw.copy(), obs_prev=obs_raw.copy(), t_remaining=0.0)
            condition = DPCondition.from_critic_state(cs_dp, advantage=1.0)
            plan_enc = self.policy.generate_plan(condition, self.norm)
            try:
                plan = encoding_to_plan(plan_enc)
            except Exception:
                env.close()
                actual_returns.append(-20.0)  # worst case
                continue

            plan_T = plan.T
            total_duration = max(3.0, plan_T + 1.0)
            n_steps = int(round(total_duration / DT))
            gains = TrackerGains()
            t = 0.0
            obs_cur = obs_raw
            episode_landed = False
            estop = False
            window_steps = max(1, int(round(0.1 / DT)))
            err_hist: list[float] = []

            for k in range(n_steps):
                obs_state = obs_to_state(obs_cur)
                action_ctrl, _ = control(obs_state, t, plan, params, gains)
                obs_next, _, terminated, truncated, _ = env.step(action_ctrl)
                t += DT
                r = plan(t)
                ns = obs_to_state(obs_next)
                e_norm = math.sqrt((ns[0]-r[0])**2 + (ns[1]-r[1])**2 + (2*(ns[4]-r[4]))**2)
                err_hist.append(e_norm)
                if len(err_hist) > window_steps:
                    err_hist.pop(0)
                if e_norm > 2.0 or (len(err_hist) == window_steps and all(e > 1.0 for e in err_hist)):
                    estop = True
                if obs_next[6] > 0.5 and obs_next[7] > 0.5:
                    episode_landed = True
                if terminated or truncated or estop or episode_landed:
                    break
                obs_cur = obs_next

            env.close()
            if episode_landed:
                actual_returns.append(-t)
            else:
                actual_returns.append(-t - 10.0)

        v_preds = np.array(v_preds)
        actual_returns = np.array(actual_returns)
        n = len(v_preds)
        if n < 2:
            print("  critic eval: insufficient data", flush=True)
            return {"corr": 0.0, "mae": 0.0}

        corr = float(np.corrcoef(v_preds, actual_returns)[0, 1])
        mae = float(np.mean(np.abs(v_preds - actual_returns)))
        bias = float(np.mean(v_preds - actual_returns))

        print(f"  critic eval ({n} seeds): corr={corr:.3f} MAE={mae:.2f} bias={bias:+.2f}", flush=True)
        print(f"    V_pred:  mean={v_preds.mean():.2f} std={v_preds.std():.2f}", flush=True)
        print(f"    actual:  mean={actual_returns.mean():.2f} std={actual_returns.std():.2f}", flush=True)
        return {"corr": corr, "mae": mae, "bias": bias}

    def _evaluate_advantage_conditioning(self, n_seeds: int = 20) -> dict[str, float]:
        """Run same seeds with advantage=-1 and +1, check for effect."""
        seeds = self.config.holdout_seeds[:n_seeds]
        results = {-1.0: [], 1.0: []}

        for adv_cond in [-1.0, 1.0]:
            for seed in seeds:
                r = run_dp_episode(
                    self.policy, self.critic, self.norm,
                    render=False, seed=seed, verbose=False,
                    advantage_cond=adv_cond,
                )
                if not r.get("planning_failed"):
                    ret = -r["t_elapsed"] if r["landed"] else -(r["t_elapsed"] + 10.0)
                    results[adv_cond].append({
                        "landed": r["landed"],
                        "return": ret,
                        "advantage": r.get("advantage", 0.0),
                    })

        pos = results[1.0]
        neg = results[-1.0]
        pos_land = sum(1 for r in pos if r["landed"]) / max(len(pos), 1)
        neg_land = sum(1 for r in neg if r["landed"]) / max(len(neg), 1)
        pos_ret = np.mean([r["return"] for r in pos]) if pos else 0.0
        neg_ret = np.mean([r["return"] for r in neg]) if neg else 0.0
        pos_adv = np.mean([r["advantage"] for r in pos]) if pos else 0.0
        neg_adv = np.mean([r["advantage"] for r in neg]) if neg else 0.0

        print(f"  advantage conditioning test ({len(seeds)} seeds):", flush=True)
        print(f"    adv=+1: land={100*pos_land:.0f}% ret={pos_ret:.2f} critic_adv={pos_adv:.3f}", flush=True)
        print(f"    adv=-1: land={100*neg_land:.0f}% ret={neg_ret:.2f} critic_adv={neg_adv:.3f}", flush=True)
        delta = pos_ret - neg_ret
        print(f"    delta(+1 minus -1): return={delta:+.2f} land={100*(pos_land-neg_land):+.0f}pp", flush=True)

        return {
            "pos_landing": pos_land, "neg_landing": neg_land,
            "pos_return": pos_ret, "neg_return": neg_ret,
            "pos_advantage": pos_adv, "neg_advantage": neg_adv,
            "delta_return": delta,
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.policy.config,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


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
) -> dict:
    import gymnasium as gym

    env = gym.make("LunarLander-v3", continuous=True,
                   render_mode="human" if render else None)
    if seed is not None:
        env.reset(seed=seed)

    try:
        obs_raw, state0, params = warmup_and_snapshot(env, n_steps=5)
    except RuntimeError as e:
        env.close()
        if verbose:
            print(f"[dp] warmup failed: {e}")
        return {"planning_failed": True, "landed": False, "t_elapsed": 0.0}

    # Generate plan with DP (t_remaining=0 for DP, 10 for critic)
    cs = CriticState(
        obs_t=obs_raw.copy(),
        obs_prev=obs_raw.copy(),
        t_remaining=10.0,
    )
    cs_dp = CriticState(
        obs_t=obs_raw.copy(),
        obs_prev=obs_raw.copy(),
        t_remaining=0.0,
    )
    condition = DPCondition.from_critic_state(cs_dp, advantage=advantage_cond)

    t0 = time.perf_counter()
    plan_enc = policy.generate_plan(condition, norm)
    planning_ms = (time.perf_counter() - t0) * 1000.0

    try:
        plan = encoding_to_plan(plan_enc)
    except Exception as e:
        env.close()
        if verbose:
            print(f"[dp] plan reconstruction failed: {e}")
        return {"planning_failed": True, "landed": False, "t_elapsed": 0.0,
                "planning_ms": planning_ms}

    # Compute advantage from critic
    critic.eval()
    with torch.no_grad():
        s_tensor = cs.to_tensor().unsqueeze(0)
        a_tensor = plan_enc.to_tensor().unsqueeze(0)
        v_val = critic.forward_v(s_tensor).item()
        q_val = critic.forward_q(s_tensor, a_tensor).item()
    advantage = q_val - v_val

    plan_T = plan.T
    total_duration = max(3.0, plan_T + 1.0)
    n_steps = int(round(total_duration / DT))
    gains = TrackerGains()

    if verbose:
        x0, y0, vx0, vy0, th0, om0 = state0
        print(f"[dp] state = x={x0:+.3f} y={y0:+.3f} vx={vx0:+.3f} vy={vy0:+.3f}")
        print(f"[dp] plan T={plan_T:.2f}s, planning={planning_ms:.1f}ms, "
              f"adv={advantage:.3f}, V={v_val:.2f}, Q={q_val:.2f}")

    errs = []
    obs_state = state0
    t = 0.0
    terminated = False
    estop = False
    landed = False
    last_obs = None

    window_steps = max(1, int(round(0.1 / DT)))
    err_hist: list[float] = []

    window_closed = False
    for k in range(n_steps):
        action, dbg = control(obs_state, t, plan, params, gains)
        obs, _, terminated, truncated, _ = env.step(action)

        if render:
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN
                        and event.key in (pygame.K_q, pygame.K_ESCAPE)
                    ):
                        window_closed = True
                draw_plan_overlay(env, plan, current_t=t)
                pygame.display.flip()
            except Exception:
                pass
            if window_closed:
                break

        obs_state = obs_to_state(obs)
        last_obs = obs
        t += DT

        r = plan(t)
        e_x = obs_state[0] - r[0]
        e_y = obs_state[1] - r[1]
        e_th = obs_state[4] - r[4]
        errs.append((e_x, e_y, e_th))

        e_norm = math.sqrt(e_x**2 + e_y**2 + (2.0 * e_th)**2)
        err_hist.append(e_norm)
        if len(err_hist) > window_steps:
            err_hist.pop(0)
        if e_norm > 2.0 or (
            len(err_hist) == window_steps and all(e > 1.0 for e in err_hist)
        ):
            estop = True
            break

        if obs[6] > 0.5 and obs[7] > 0.5:
            landed = True
            break

        if terminated or truncated:
            break

    env.close()
    errs_arr = np.array(errs) if errs else np.zeros((0, 3))
    final_state = obs_to_state(last_obs) if last_obs is not None else state0

    metrics = dict(
        steps=len(errs_arr),
        terminated=terminated,
        estop=estop,
        landed=landed,
        success=bool(landed),
        window_closed=window_closed,
        planning_failed=False,
        planning_ms=planning_ms,
        advantage=advantage,
        t_elapsed=t,
        final_x=float(final_state[0]),
        final_y=float(final_state[1]),
        max_abs_x_err=float(np.max(np.abs(errs_arr[:, 0]))) if len(errs_arr) else 0.0,
        max_abs_y_err=float(np.max(np.abs(errs_arr[:, 1]))) if len(errs_arr) else 0.0,
        max_abs_th_err=float(np.max(np.abs(errs_arr[:, 2]))) if len(errs_arr) else 0.0,
        rms_x_err=float(np.sqrt(np.mean(errs_arr[:, 0] ** 2))) if len(errs_arr) else 0.0,
        rms_y_err=float(np.sqrt(np.mean(errs_arr[:, 1] ** 2))) if len(errs_arr) else 0.0,
        rms_th_err=float(np.sqrt(np.mean(errs_arr[:, 2] ** 2))) if len(errs_arr) else 0.0,
    )

    if verbose:
        status = ("LANDED" if landed else
                  "ESTOP " if estop else
                  "CRASH " if terminated else
                  "TIMED ")
        print(f"[dp] {status} steps={metrics['steps']}/{n_steps} "
              f"final=({final_state[0]:+.2f}, {final_state[1]:+.2f})")

    return metrics


# ===========================================================================
# CLI
# ===========================================================================

def _make_pickle_helper(module):
    """Create a pickle-compatible module shim so torch.load can find
    classes like CriticConfig/TrunkType when __main__ is diffusion_policy."""
    import pickle as _pickle
    import types

    class _Unpickler(_pickle.Unpickler):
        def find_class(self, mod_name, name):
            if mod_name == "__main__" and hasattr(module, name):
                return getattr(module, name)
            return super().find_class(mod_name, name)

    # torch.load expects a module with Unpickler + load/loads
    shim = types.ModuleType("_pickle_shim")
    shim.Unpickler = _Unpickler
    shim.load = _pickle.load
    shim.loads = _pickle.loads
    return shim


def main():
    parser = argparse.ArgumentParser(description="Diffusion policy: phase2, phase3, evaluate, run, compare")
    sub = parser.add_subparsers(dest="cmd")

    # phase2
    p2 = sub.add_parser("phase2", help="Behavioral cloning from KTO episodes")
    p2.add_argument("--db", type=str, default="dp_rollout_db")
    p2.add_argument("--source-db", type=str, default="rollout_db",
                     help="Existing RolloutDB to upgrade from (if dp DB doesn't exist)")
    p2.add_argument("--critic-ckpt", type=str, default="checkpoints/best.pt")
    p2.add_argument("--epochs", type=int, default=100)
    p2.add_argument("--lr", type=float, default=3e-4)
    p2.add_argument("--collect", type=int, default=0,
                     help="Collect N additional KTO episodes before training")

    # phase3
    p3 = sub.add_parser("phase3", help="Online improvement")
    p3.add_argument("--db", type=str, default="dp_rollout_db")
    p3.add_argument("--dp-ckpt", type=str, default="dp_checkpoints/best.pt")
    p3.add_argument("--critic-ckpt", type=str, default="checkpoints/best.pt")
    p3.add_argument("--iterations", type=int, default=20)
    p3.add_argument("--episodes-per-iter", type=int, default=500)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate on holdout seeds")
    p_eval.add_argument("--dp-ckpt", type=str, default="dp_checkpoints/best.pt")
    p_eval.add_argument("--critic-ckpt", type=str, default="checkpoints/best.pt")
    p_eval.add_argument("--norm", type=str, default="dp_checkpoints/norm.npz")

    # run
    p_run = sub.add_parser("run", help="Run single DP episode")
    p_run.add_argument("--dp-ckpt", type=str, default="dp_checkpoints/best.pt")
    p_run.add_argument("--critic-ckpt", type=str, default="checkpoints/best.pt")
    p_run.add_argument("--norm", type=str, default="dp_checkpoints/norm.npz")
    p_run.add_argument("--render", action="store_true")
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--advantage", type=float, default=1.0)

    # adv-test
    # critic-eval
    p_crit = sub.add_parser("critic-eval", help="Evaluate critic V(s) vs actual returns")
    p_crit.add_argument("--dp-ckpt", type=str, default="dp_checkpoints/best.pt")
    p_crit.add_argument("--critic-ckpt", type=str, default="checkpoints/best.pt")
    p_crit.add_argument("--norm", type=str, default="dp_checkpoints/norm.npz")
    p_crit.add_argument("--seeds", type=int, default=50)

    p_adv = sub.add_parser("adv-test", help="Test advantage conditioning effect")
    p_adv.add_argument("--dp-ckpt", type=str, default="dp_checkpoints/best.pt")
    p_adv.add_argument("--critic-ckpt", type=str, default="checkpoints/best.pt")
    p_adv.add_argument("--norm", type=str, default="dp_checkpoints/norm.npz")
    p_adv.add_argument("--seeds", type=int, default=20)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare KTO vs DP on same seeds")
    p_cmp.add_argument("--dp-ckpt", type=str, default="dp_checkpoints/best.pt")
    p_cmp.add_argument("--critic-ckpt", type=str, default="checkpoints/best.pt")
    p_cmp.add_argument("--norm", type=str, default="dp_checkpoints/norm.npz")
    p_cmp.add_argument("--episodes", type=int, default=100)

    args = parser.parse_args()

    def _load_critic(ckpt_path: str) -> LanderCritic:
        import lander_critic as _lc
        ckpt = torch.load(Path(ckpt_path), weights_only=False,
                          pickle_module=_make_pickle_helper(_lc))
        critic = LanderCritic(ckpt["config"])
        critic.load_state_dict(ckpt["critic"])
        critic.eval()
        return critic

    def _load_dp(ckpt_path: str) -> DiffusionPolicy:
        ckpt = torch.load(Path(ckpt_path), weights_only=False)
        policy = DiffusionPolicy(ckpt["config"])
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        return policy

    if args.cmd == "phase2":
        # Load or create DB
        db_path = Path(args.db)
        if not (db_path / "meta.json").exists():
            source_path = Path(args.source_db)
            if (source_path / "meta.json").exists():
                print(f"[phase2] upgrading {source_path} -> {db_path}")
                source_db = RolloutDB(source_path, readonly=True)
                db = DPRolloutDB.upgrade_from(source_db, db_path)
            else:
                print(f"[phase2] creating new DB at {db_path}")
                db = DPRolloutDB.create(db_path)
        else:
            db = DPRolloutDB(db_path)

        # Optionally collect more KTO episodes
        if args.collect > 0:
            from lander_critic import collect_rollouts
            print(f"[phase2] collecting {args.collect} KTO episodes...")
            collect_rollouts(db, n_episodes=args.collect)

        # Load critic
        critic = _load_critic(args.critic_ckpt)

        # Compute normalization
        norm = PlanNormalization.fit(db)
        norm_path = Path("dp_checkpoints") / "norm.npz"
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        norm.save(norm_path)
        print(f"[phase2] normalization saved to {norm_path}")

        # Build policy
        dp_cfg = DPConfig()
        policy = DiffusionPolicy(dp_cfg)
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"[phase2] model: {n_params:,} params")

        train_cfg = DPTrainConfig(phase=2, epochs=args.epochs, lr=args.lr,
                                   advantage_dropout=1.0)  # unconditional BC
        trainer = DPTrainer(policy, critic, db, norm, train_cfg)
        result = trainer.train_phase2()
        print(f"[phase2] result: landing_rate={result.landing_rate:.1%}, "
              f"mean_adv={result.mean_advantage:.3f}")

    elif args.cmd == "phase3":
        db = DPRolloutDB(Path(args.db))
        critic = _load_critic(args.critic_ckpt)
        policy = _load_dp(args.dp_ckpt)
        norm = PlanNormalization.load(Path("dp_checkpoints/norm.npz"))

        train_cfg = DPTrainConfig(
            phase=3,
            n_iterations=args.iterations,
            episodes_per_iteration=args.episodes_per_iter,
        )
        trainer = DPTrainer(policy, critic, db, norm, train_cfg)
        result = trainer.train_phase3()
        print(f"[phase3] result: landing_rate={result.landing_rate:.1%}, "
              f"mean_adv={result.mean_advantage:.3f}")

    elif args.cmd == "evaluate":
        critic = _load_critic(args.critic_ckpt)
        policy = _load_dp(args.dp_ckpt)
        norm = PlanNormalization.load(Path(args.norm))

        seeds = list(range(5000, 5100))
        results = []
        for seed in seeds:
            r = run_dp_episode(policy, critic, norm, seed=seed, verbose=False)
            results.append(r)

        landed = sum(1 for r in results if r.get("landed"))
        valid = [r for r in results if not r.get("planning_failed")]
        mean_return = np.mean([
            -r["t_elapsed"] if r["landed"] else -(r["t_elapsed"] + 10.0)
            for r in valid
        ]) if valid else 0.0
        mean_adv = np.mean([r.get("advantage", 0) for r in valid]) if valid else 0.0
        mean_latency = np.mean([r.get("planning_ms", 0) for r in valid]) if valid else 0.0

        print(f"\n{'='*60}")
        print(f"DP EVALUATION over {len(seeds)} holdout seeds")
        print(f"{'='*60}")
        print(f"  Landing rate:    {landed}/{len(valid)} ({100*landed/max(len(valid),1):.0f}%)")
        print(f"  Mean return:     {mean_return:.2f}")
        print(f"  Mean advantage:  {mean_adv:.3f}")
        print(f"  Planning time:   {mean_latency:.1f} ms")

    elif args.cmd == "critic-eval":
        critic = _load_critic(args.critic_ckpt)
        policy = _load_dp(args.dp_ckpt)
        norm = PlanNormalization.load(Path(args.norm))

        db = DPRolloutDB(Path("dp_rollout_db"), readonly=True) if Path("dp_rollout_db/meta.json").exists() else None
        train_cfg = DPTrainConfig(holdout_seeds=list(range(5000, 5000 + args.seeds)))
        trainer = DPTrainer(policy, critic, db, norm, train_cfg)
        trainer._evaluate_critic(n_seeds=args.seeds)

    elif args.cmd == "adv-test":
        critic = _load_critic(args.critic_ckpt)
        policy = _load_dp(args.dp_ckpt)
        norm = PlanNormalization.load(Path(args.norm))

        db = DPRolloutDB(Path("dp_rollout_db"), readonly=True) if Path("dp_rollout_db/meta.json").exists() else None
        train_cfg = DPTrainConfig(holdout_seeds=list(range(5000, 5000 + args.seeds)))
        trainer = DPTrainer(policy, critic, db, norm, train_cfg)
        trainer._evaluate_advantage_conditioning(n_seeds=args.seeds)

    elif args.cmd == "run":
        critic = _load_critic(args.critic_ckpt)
        policy = _load_dp(args.dp_ckpt)
        norm = PlanNormalization.load(Path(args.norm))

        if args.render:
            # Loop until window closed
            seed = args.seed or 0
            while True:
                m = run_dp_episode(
                    policy, critic, norm,
                    render=True, seed=seed, advantage_cond=args.advantage,
                )
                if m.get("window_closed"):
                    break
                seed += 1
        else:
            run_dp_episode(
                policy, critic, norm,
                seed=args.seed, advantage_cond=args.advantage,
            )

    elif args.cmd == "compare":
        from kto_lander import run_episode as run_kto_episode

        critic = _load_critic(args.critic_ckpt)
        policy = _load_dp(args.dp_ckpt)
        norm = PlanNormalization.load(Path(args.norm))

        seeds = list(range(5000, 5000 + args.episodes))
        kto_results = []
        dp_results = []

        print(f"Running {args.episodes} episodes each for KTO and DP...")
        for seed in seeds:
            try:
                kto_r = run_kto_episode(render=False, seed=seed, verbose=False)
                kto_results.append(kto_r)
            except Exception:
                kto_results.append(None)

            dp_r = run_dp_episode(policy, critic, norm, seed=seed, verbose=False)
            dp_results.append(dp_r)

        # Summarize
        kto_valid = [r for r in kto_results if r is not None]
        dp_valid = [r for r in dp_results if not r.get("planning_failed")]

        kto_landed = sum(1 for r in kto_valid if r["landed"])
        dp_landed = sum(1 for r in dp_valid if r.get("landed"))

        def _return(r, is_kto=False):
            if is_kto:
                t = r["steps"] * DT
                return -t if r["landed"] else -(t + 10.0)
            else:
                return -r["t_elapsed"] if r["landed"] else -(r["t_elapsed"] + 10.0)

        kto_returns = [_return(r, True) for r in kto_valid]
        dp_returns = [_return(r, False) for r in dp_valid]

        print(f"\n{'='*60}")
        print(f"COMPARISON: KTO vs DP ({args.episodes} seeds)")
        print(f"{'='*60}")
        print(f"  {'Metric':<25} {'KTO':>10} {'DP':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10}")
        print(f"  {'Landing rate':<25} {kto_landed}/{len(kto_valid):>5} {dp_landed}/{len(dp_valid):>5}")
        print(f"  {'Landing %':<25} {100*kto_landed/max(len(kto_valid),1):>9.0f}% {100*dp_landed/max(len(dp_valid),1):>9.0f}%")
        if kto_returns:
            print(f"  {'Mean return':<25} {np.mean(kto_returns):>10.2f} {np.mean(dp_returns):>10.2f}")
        dp_latency = np.mean([r.get("planning_ms", 0) for r in dp_valid])
        print(f"  {'Planning time (ms)':<25} {'~100':>10} {dp_latency:>10.1f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
