"""Learned V/Q critic for KTO lunar lander.

Shared-trunk architecture with dual heads:
  - V(s): value of state s = (obs_t, obs_prev, t_remaining)
  - Q(s, a): value of state s under plan a (encoded spline control points + T)

Reward:
  - r = 0              non-terminal steps
  - r = -t_elapsed     terminal, successful landing
  - r = -t_elapsed-10  terminal, failure (crash / timeout / e-stop)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kto_lander import Plan

# ===========================================================================
# Constants
# ===========================================================================

STATE_DIM = 17       # obs_t(8) + obs_prev(8) + t_remaining(1)
DEFAULT_NUM_CONTROL_POINTS = 12
ACTION_DIM = 25      # 12*2 + 1 (flattened control points + duration)


# ===========================================================================
# State and action encoding
# ===========================================================================

@dataclass
class CriticState:
    obs_t: np.ndarray         # (8,)
    obs_prev: np.ndarray      # (8,)
    t_remaining: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            np.concatenate([self.obs_t, self.obs_prev, [self.t_remaining]]),
            dtype=torch.float32,
        )

    @staticmethod
    def from_tensor(t: torch.Tensor) -> CriticState:
        a = t.detach().cpu().numpy()
        return CriticState(obs_t=a[:8], obs_prev=a[8:16], t_remaining=float(a[16]))


@dataclass
class PlanEncoding:
    control_points: np.ndarray   # (num_cp * 2,)
    duration: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            np.concatenate([self.control_points, [self.duration]]),
            dtype=torch.float32,
        )

    @staticmethod
    def from_plan(plan: Plan) -> PlanEncoding:
        cps = []
        for cp in plan.traj.control_points():
            arr = np.asarray(cp).flatten()
            cps.extend(arr[:2].tolist())
        return PlanEncoding(
            control_points=np.array(cps, dtype=np.float64),
            duration=plan.T,
        )


# ===========================================================================
# Reward
# ===========================================================================

class Outcome(Enum):
    IN_PROGRESS = 0
    LANDED = 1
    FAILED = 2


def reward(outcome: Outcome, t_elapsed: float) -> float:
    if outcome == Outcome.IN_PROGRESS:
        return 0.0
    elif outcome == Outcome.LANDED:
        return -t_elapsed
    else:  # FAILED
        return -t_elapsed - 10.0


# ===========================================================================
# Model
# ===========================================================================

class TrunkType(Enum):
    MLP = "mlp"
    TRANSFORMER = "transformer"


@dataclass
class CriticConfig:
    trunk_type: TrunkType = TrunkType.MLP
    trunk_hidden: int = 320
    trunk_layers: int = 4
    head_hidden: int = 128
    head_layers: int = 2
    n_heads: int = 4       # transformer only
    dropout: float = 0.0

    @property
    def param_count_estimate(self) -> int:
        if self.trunk_type == TrunkType.MLP:
            # state_encoder: STATE_DIM -> trunk_hidden
            p = STATE_DIM * self.trunk_hidden + self.trunk_hidden
            # trunk layers
            for _ in range(self.trunk_layers):
                p += self.trunk_hidden * self.trunk_hidden + self.trunk_hidden
            # V head
            v_in = self.trunk_hidden
            for _ in range(self.head_layers - 1):
                p += v_in * self.head_hidden + self.head_hidden
                v_in = self.head_hidden
            p += v_in * 1 + 1
            # Q head (gets trunk_hidden + ACTION_DIM)
            q_in = self.trunk_hidden + ACTION_DIM
            for i in range(self.head_layers - 1):
                out = self.head_hidden
                p += q_in * out + out
                q_in = out
            p += q_in * 1 + 1
            return p
        else:
            # rough transformer estimate
            d = self.trunk_hidden
            p = STATE_DIM * d + d  # encoder
            # each transformer layer: ~4*d^2 (attn) + 2*d*4d (ffn)
            p += self.trunk_layers * (4 * d * d + 2 * d * 4 * d)
            # heads (same as MLP)
            p += (self.head_layers * self.head_hidden * self.head_hidden +
                  self.head_hidden + 1) * 2
            return p


class _MLPTrunk(nn.Module):
    def __init__(self, cfg: CriticConfig):
        super().__init__()
        layers = [nn.Linear(STATE_DIM, cfg.trunk_hidden), nn.ReLU()]
        for _ in range(cfg.trunk_layers):
            layers.append(nn.Linear(cfg.trunk_hidden, cfg.trunk_hidden))
            layers.append(nn.LayerNorm(cfg.trunk_hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TransformerTrunk(nn.Module):
    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.encoder = nn.Linear(STATE_DIM, cfg.trunk_hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.trunk_hidden,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.trunk_hidden * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.trunk_layers)
        self.ln = nn.LayerNorm(cfg.trunk_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, STATE_DIM) -> treat as single-token sequence
        h = self.encoder(x).unsqueeze(1)  # (batch, 1, d)
        h = self.transformer(h)
        h = self.ln(h.squeeze(1))  # (batch, d)
        return h


def _build_head(in_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers.extend([nn.Linear(d, hidden), nn.ReLU()])
        d = hidden
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class LanderCritic(nn.Module):
    def __init__(self, config: CriticConfig | None = None):
        super().__init__()
        self.config = config or CriticConfig()
        cfg = self.config

        if cfg.trunk_type == TrunkType.MLP:
            self.trunk = _MLPTrunk(cfg)
        else:
            self.trunk = _TransformerTrunk(cfg)

        self.v_head = _build_head(cfg.trunk_hidden, cfg.head_hidden, cfg.head_layers)
        self.q_head = _build_head(
            cfg.trunk_hidden + ACTION_DIM, cfg.head_hidden, cfg.head_layers
        )

    def forward_v(self, state: torch.Tensor) -> torch.Tensor:
        h = self.trunk(state)
        return self.v_head(h)

    def forward_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self.trunk(state)
        ha = torch.cat([h, action], dim=-1)
        return self.q_head(ha)

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        if action is None:
            return self.forward_v(state)
        return self.forward_q(state, action)


# ===========================================================================
# Rollout database
# ===========================================================================

@dataclass
class Transition:
    state: CriticState
    action: PlanEncoding | None
    reward: float
    next_state: CriticState | None
    outcome: Outcome
    t_elapsed: float
    episode_id: int
    step_idx: int


class RolloutDB:
    """On-disk rollout database backed by numpy arrays."""

    def __init__(self, db_path: Path, readonly: bool = False):
        self.db_path = Path(db_path)
        self.readonly = readonly
        meta_path = self.db_path / "meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)
        self._size = self.meta["size"]
        self._num_episodes = self.meta["num_episodes"]
        self._capacity = self.meta["capacity"]
        mode = "r" if readonly else "r+"
        self._states = np.load(self.db_path / "states.npy", mmap_mode=mode)
        self._actions = np.load(self.db_path / "actions.npy", mmap_mode=mode)
        self._rewards = np.load(self.db_path / "rewards.npy", mmap_mode=mode)
        mc_path = self.db_path / "mc_returns.npy"
        if mc_path.exists():
            self._mc_returns = np.load(mc_path, mmap_mode=mode)
        else:
            self._mc_returns = None
        self._next_states = np.load(self.db_path / "next_states.npy", mmap_mode=mode)
        self._terminals = np.load(self.db_path / "terminals.npy", mmap_mode=mode)
        self._episode_ids = np.load(self.db_path / "episode_ids.npy", mmap_mode=mode)
        self._step_indices = np.load(self.db_path / "step_indices.npy", mmap_mode=mode)

    @staticmethod
    def create(db_path: Path, capacity: int = 100_000) -> RolloutDB:
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
        meta = {"capacity": capacity, "size": 0, "num_episodes": 0, "version": 1}
        with open(db_path / "meta.json", "w") as f:
            json.dump(meta, f)
        return RolloutDB(db_path)

    def append_episode(self, transitions: Sequence[Transition]) -> int:
        if self.readonly:
            raise RuntimeError("DB is readonly")
        ep_id = self._num_episodes
        # With gamma=1 and sparse reward (0 except terminal), MC return for every
        # step in the episode equals the terminal reward.
        terminal_reward = 0.0
        for tr in transitions:
            if tr.outcome != Outcome.IN_PROGRESS:
                terminal_reward = tr.reward
        start_idx = self._size
        for tr in transitions:
            if self._size >= self._capacity:
                raise RuntimeError("DB full")
            i = self._size
            self._states[i] = tr.state.to_tensor().numpy()
            if tr.action is not None:
                self._actions[i] = tr.action.to_tensor().numpy()
            self._rewards[i] = tr.reward
            if self._mc_returns is not None:
                self._mc_returns[i] = terminal_reward
            if tr.next_state is not None:
                self._next_states[i] = tr.next_state.to_tensor().numpy()
            self._terminals[i] = tr.outcome != Outcome.IN_PROGRESS
            self._episode_ids[i] = ep_id
            self._step_indices[i] = tr.step_idx
            self._size += 1

        self._num_episodes += 1
        # flush meta
        self.meta["size"] = self._size
        self.meta["num_episodes"] = self._num_episodes
        with open(self.db_path / "meta.json", "w") as f:
            json.dump(self.meta, f)
        return ep_id

    def sample_batch(
        self, batch_size: int, terminal_only: bool = False
    ) -> dict[str, torch.Tensor]:
        if terminal_only:
            indices = np.where(self._terminals[:self._size])[0]
        else:
            indices = np.arange(self._size)
        chosen = np.random.choice(indices, size=min(batch_size, len(indices)), replace=False)
        batch = {
            "state": torch.tensor(self._states[chosen], dtype=torch.float32),
            "action": torch.tensor(self._actions[chosen], dtype=torch.float32),
            "reward": torch.tensor(self._rewards[chosen], dtype=torch.float32),
            "next_state": torch.tensor(self._next_states[chosen], dtype=torch.float32),
            "terminal": torch.tensor(self._terminals[chosen], dtype=torch.bool),
        }
        if self._mc_returns is not None:
            batch["mc_return"] = torch.tensor(
                self._mc_returns[chosen], dtype=torch.float32
            )
        return batch

    def __len__(self) -> int:
        return self._size

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    def episode_returns(self) -> np.ndarray:
        returns = []
        for ep in range(self._num_episodes):
            mask = self._episode_ids[:self._size] == ep
            returns.append(float(self._rewards[:self._size][mask].sum()))
        return np.array(returns)


# ===========================================================================
# Rollout collector
# ===========================================================================

def collect_rollouts(
    db: RolloutDB,
    n_episodes: int = 100,
    duration_s: float = 3.0,
    verbose: bool = True,
) -> dict[str, float]:
    """Run n_episodes of kto_lander and store transitions in db.

    Returns summary stats.
    """
    import gymnasium as gym
    from kto_lander import (
        DT, TrackerGains, control, obs_to_state, plan_with_kto,
        warmup_and_snapshot,
    )

    landed_count = 0
    failed_count = 0
    solver_fail_count = 0

    for ep in range(n_episodes):
        seed = ep + 1000  # deterministic but different from any prior runs
        env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
        env.reset(seed=seed)

        try:
            obs_raw, state0, params = warmup_and_snapshot(env, n_steps=5)
        except RuntimeError:
            env.close()
            solver_fail_count += 1
            continue

        x0, y0, vx0, vy0 = state0[0], state0[1], state0[2], state0[3]
        try:
            plan, plan_T, _ = plan_with_kto(x0, y0, vx0, vy0, params, verbose=False)
        except RuntimeError:
            env.close()
            solver_fail_count += 1
            continue

        plan_enc = PlanEncoding.from_plan(plan)
        total_duration = max(duration_s, plan_T + 1.0)
        n_steps = int(round(total_duration / DT))
        gains = TrackerGains()

        transitions = []
        obs_prev = obs_raw.copy()
        obs_cur = obs_raw
        t = 0.0
        terminated = False
        episode_landed = False
        estop = False

        # E-stop tracking (matches kto_lander logic)
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

            action_ctrl, dbg = control(obs_state, t, plan, params, gains)
            obs_next, _, terminated, truncated, info = env.step(action_ctrl)
            t += DT

            # E-stop check
            r = plan(t)
            next_state_si = obs_to_state(obs_next)
            e_x = next_state_si[0] - r[0]
            e_y = next_state_si[1] - r[1]
            e_th = next_state_si[4] - r[4]
            e_norm = math.sqrt(e_x**2 + e_y**2 + (2.0 * e_th)**2)
            err_hist.append(e_norm)
            if len(err_hist) > window_steps:
                err_hist.pop(0)
            if e_norm > 2.0 or (
                len(err_hist) == window_steps and all(e > 1.0 for e in err_hist)
            ):
                estop = True

            # Landing check
            if obs_next[6] > 0.5 and obs_next[7] > 0.5:
                episode_landed = True

            is_terminal = terminated or truncated or estop or episode_landed

            if is_terminal:
                if episode_landed:
                    outcome = Outcome.LANDED
                else:
                    outcome = Outcome.FAILED
            else:
                outcome = Outcome.IN_PROGRESS

            r_val = reward(outcome, t)

            next_critic_state = CriticState(
                obs_t=obs_next.copy(),
                obs_prev=obs_cur.copy(),
                t_remaining=max(0.0, total_duration - t),
            )

            transitions.append(Transition(
                state=critic_state,
                action=plan_enc,
                reward=r_val,
                next_state=None if is_terminal else next_critic_state,
                outcome=outcome,
                t_elapsed=t,
                episode_id=0,  # will be set by append_episode
                step_idx=k,
            ))

            obs_prev = obs_cur.copy()
            obs_cur = obs_next

            if is_terminal:
                break

        env.close()
        db.append_episode(transitions)

        if episode_landed:
            landed_count += 1
        else:
            failed_count += 1

        if verbose and (ep + 1) % 10 == 0:
            print(f"  collected {ep+1}/{n_episodes} episodes "
                  f"(landed={landed_count}, failed={failed_count}, "
                  f"solver_fail={solver_fail_count})")

    stats = {
        "total": n_episodes,
        "landed": landed_count,
        "failed": failed_count,
        "solver_failures": solver_fail_count,
        "transitions": len(db),
    }
    if verbose:
        print(f"[collect] done: {stats}")
    return stats


# ===========================================================================
# Training
# ===========================================================================

@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 1.0          # undiscounted — reward is already time
    tau: float = 0.005           # target net soft-update rate
    v_coeff: float = 1.0
    q_coeff: float = 1.0
    max_grad_norm: float = 1.0
    epochs: int = 100
    steps_per_epoch: int = 100
    val_frac: float = 0.1
    checkpoint_dir: Path = Path("checkpoints")


@dataclass
class TrainResult:
    v_loss_final: float
    q_loss_final: float
    v_loss_history: list[float]
    q_loss_history: list[float]
    val_v_loss: float
    val_q_loss: float
    total_steps: int
    best_epoch: int


class TrainCallback:
    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        pass

    def on_batch_end(self, step: int, loss: float) -> None:
        pass


class Trainer:
    """Trains LanderCritic using Monte Carlo returns (exact for sparse reward).

    V loss:  MSE( V(s), G )     where G = MC return for the episode
    Q loss:  MSE( Q(s,a), G )

    With gamma=1 and reward=0 except terminal, G is constant across all steps
    in an episode: G = -t_landed or G = -(t_failed + 10).
    """

    def __init__(
        self,
        critic: LanderCritic,
        db: RolloutDB,
        config: TrainConfig | None = None,
    ):
        self.critic = critic
        self.db = db
        self.config = config or TrainConfig()
        self.optimizer = torch.optim.Adam(critic.parameters(), lr=self.config.lr)

    def train(self, callback: TrainCallback | None = None) -> TrainResult:
        cfg = self.config
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        v_loss_hist = []
        q_loss_hist = []
        best_val = float("inf")
        best_epoch = 0

        for epoch in range(cfg.epochs):
            self.critic.train()
            epoch_v_loss = 0.0
            epoch_q_loss = 0.0

            for step in range(cfg.steps_per_epoch):
                batch = self.db.sample_batch(cfg.batch_size)
                states = batch["state"]
                actions = batch["action"]
                mc_target = batch["mc_return"]

                # V loss
                v_pred = self.critic.forward_v(states).squeeze(-1)
                v_loss = F.mse_loss(v_pred, mc_target)

                # Q loss — only on transitions with valid actions (non-NaN)
                has_action = ~torch.isnan(actions[:, 0])
                if has_action.any():
                    q_pred = self.critic.forward_q(
                        states[has_action], actions[has_action]
                    ).squeeze(-1)
                    q_loss = F.mse_loss(q_pred, mc_target[has_action])
                else:
                    q_loss = torch.tensor(0.0)

                loss = cfg.v_coeff * v_loss + cfg.q_coeff * q_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                epoch_v_loss += v_loss.item()
                epoch_q_loss += q_loss.item()

                if callback:
                    callback.on_batch_end(step, loss.item())

            avg_v = epoch_v_loss / cfg.steps_per_epoch
            avg_q = epoch_q_loss / cfg.steps_per_epoch
            v_loss_hist.append(avg_v)
            q_loss_hist.append(avg_q)

            # Validation
            val_batch = self.db.sample_batch(min(1024, len(self.db)))
            self.critic.eval()
            with torch.no_grad():
                val_mc = val_batch["mc_return"]
                val_v = F.mse_loss(
                    self.critic.forward_v(val_batch["state"]).squeeze(-1), val_mc
                ).item()
                va = val_batch["action"]
                ha = ~torch.isnan(va[:, 0])
                if ha.any():
                    val_q = F.mse_loss(
                        self.critic.forward_q(
                            val_batch["state"][ha], va[ha]
                        ).squeeze(-1),
                        val_mc[ha],
                    ).item()
                else:
                    val_q = 0.0

            if val_v + val_q < best_val:
                best_val = val_v + val_q
                best_epoch = epoch
                self.save_checkpoint(cfg.checkpoint_dir / "best.pt")

            if callback:
                callback.on_epoch_end(epoch, {
                    "v_loss": avg_v, "q_loss": avg_q,
                    "val_v_loss": val_v, "val_q_loss": val_q,
                })

            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{cfg.epochs}  "
                      f"V={avg_v:.4f}  Q={avg_q:.4f}  "
                      f"val_V={val_v:.4f}  val_Q={val_q:.4f}")

        return TrainResult(
            v_loss_final=v_loss_hist[-1],
            q_loss_final=q_loss_hist[-1],
            v_loss_history=v_loss_hist,
            q_loss_history=q_loss_hist,
            val_v_loss=val_v,
            val_q_loss=val_q,
            total_steps=cfg.epochs * cfg.steps_per_epoch,
            best_epoch=best_epoch,
        )

    def save_checkpoint(self, path: Path) -> None:
        torch.save({
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.critic.config,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, weights_only=False)
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


# ===========================================================================
# Inference helpers
# ===========================================================================

def rank_plans(
    critic: LanderCritic,
    state: CriticState,
    plans: Sequence[Plan],
) -> list[tuple[float, Plan]]:
    critic.eval()
    s = state.to_tensor().unsqueeze(0)
    results = []
    with torch.no_grad():
        for plan in plans:
            a = PlanEncoding.from_plan(plan).to_tensor().unsqueeze(0)
            q = critic.forward_q(s, a).item()
            results.append((q, plan))
    results.sort(key=lambda x: x[0], reverse=True)  # higher (less negative) is better
    return results


def is_state_feasible(
    critic: LanderCritic,
    state: CriticState,
    threshold: float = -15.0,
) -> bool:
    critic.eval()
    s = state.to_tensor().unsqueeze(0)
    with torch.no_grad():
        v = critic.forward_v(s).item()
    return v > threshold


# ===========================================================================
# CLI: collect, train, validate
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lander critic: collect, train, validate")
    sub = parser.add_subparsers(dest="cmd")

    # collect
    p_collect = sub.add_parser("collect", help="Collect rollouts into DB")
    p_collect.add_argument("--db", type=str, default="rollout_db")
    p_collect.add_argument("--episodes", type=int, default=200)
    p_collect.add_argument("--capacity", type=int, default=200_000)

    # train
    p_train = sub.add_parser("train", help="Train critic from DB")
    p_train.add_argument("--db", type=str, default="rollout_db")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--trunk", choices=["mlp", "transformer"], default="mlp")
    p_train.add_argument("--lr", type=float, default=3e-4)

    # validate
    p_val = sub.add_parser("validate", help="Validate critic predictions against held-out episodes")
    p_val.add_argument("--db", type=str, default="rollout_db")
    p_val.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p_val.add_argument("--episodes", type=int, default=50)

    args = parser.parse_args()

    if args.cmd == "collect":
        db_path = Path(args.db)
        if not (db_path / "meta.json").exists():
            print(f"[collect] creating new DB at {db_path} (capacity={args.capacity})")
            db = RolloutDB.create(db_path, capacity=args.capacity)
        else:
            db = RolloutDB(db_path)
            print(f"[collect] appending to existing DB ({len(db)} transitions, "
                  f"{db.num_episodes} episodes)")
        collect_rollouts(db, n_episodes=args.episodes)

    elif args.cmd == "train":
        db = RolloutDB(Path(args.db), readonly=True)
        print(f"[train] DB has {len(db)} transitions, {db.num_episodes} episodes")
        trunk = TrunkType.MLP if args.trunk == "mlp" else TrunkType.TRANSFORMER
        cfg = CriticConfig(trunk_type=trunk)
        critic = LanderCritic(cfg)
        n_params = sum(p.numel() for p in critic.parameters())
        print(f"[train] model: {cfg.trunk_type.value}, {n_params:,} params")

        train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr)
        trainer = Trainer(critic, db, train_cfg)
        result = trainer.train()
        print(f"[train] done — best epoch {result.best_epoch}, "
              f"val_V={result.val_v_loss:.4f}, val_Q={result.val_q_loss:.4f}")

    elif args.cmd == "validate":
        import gymnasium as gym
        from kto_lander import (
            DT, TrackerGains, control, obs_to_state, plan_with_kto,
            warmup_and_snapshot,
        )

        ckpt = torch.load(Path(args.checkpoint), weights_only=False)
        cfg = ckpt["config"]
        critic = LanderCritic(cfg)
        critic.load_state_dict(ckpt["critic"])
        critic.eval()

        v_preds = []
        actual_returns = []
        outcomes = []

        for ep in range(args.episodes):
            seed = ep + 5000  # seeds not in training set
            env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
            env.reset(seed=seed)
            try:
                obs_raw, state0, params = warmup_and_snapshot(env, n_steps=5)
                plan, plan_T, _ = plan_with_kto(
                    state0[0], state0[1], state0[2], state0[3], params, verbose=False
                )
            except RuntimeError:
                env.close()
                continue

            # Get V prediction at initial state
            cs = CriticState(
                obs_t=obs_raw.copy(),
                obs_prev=obs_raw.copy(),
                t_remaining=max(3.0, plan_T + 1.0),
            )
            with torch.no_grad():
                v = critic.forward_v(cs.to_tensor().unsqueeze(0)).item()
            v_preds.append(v)

            # Roll out to get actual return
            total_duration = max(3.0, plan_T + 1.0)
            n_steps = int(round(total_duration / DT))
            gains = TrackerGains()
            t = 0.0
            obs_cur = obs_raw
            ep_return = 0.0
            ep_landed = False

            window_steps = max(1, int(round(0.1 / DT)))
            err_hist: list[float] = []

            for k in range(n_steps):
                obs_state = obs_to_state(obs_cur)
                action_ctrl, _ = control(obs_state, t, plan, params, gains)
                obs_next, _, terminated, truncated, _ = env.step(action_ctrl)
                t += DT

                r = plan(t)
                ns = obs_to_state(obs_next)
                e_x = ns[0] - r[0]
                e_y = ns[1] - r[1]
                e_th = ns[4] - r[4]
                e_norm = math.sqrt(e_x**2 + e_y**2 + (2.0 * e_th)**2)
                err_hist.append(e_norm)
                if len(err_hist) > window_steps:
                    err_hist.pop(0)
                estop = e_norm > 2.0 or (
                    len(err_hist) == window_steps and all(e > 1.0 for e in err_hist)
                )

                if obs_next[6] > 0.5 and obs_next[7] > 0.5:
                    ep_landed = True

                is_terminal = terminated or truncated or estop or ep_landed
                if is_terminal:
                    if ep_landed:
                        ep_return = -t
                    else:
                        ep_return = -t - 10.0
                    break

                obs_cur = obs_next

            env.close()
            actual_returns.append(ep_return)
            outcomes.append("landed" if ep_landed else "failed")

        v_preds = np.array(v_preds)
        actual_returns = np.array(actual_returns)

        # Correlation
        if len(v_preds) > 1:
            corr = np.corrcoef(v_preds, actual_returns)[0, 1]
        else:
            corr = 0.0
        mae = np.mean(np.abs(v_preds - actual_returns))
        landed_mask = np.array([o == "landed" for o in outcomes])

        print(f"\n{'='*60}")
        print(f"VALIDATION over {len(v_preds)} episodes")
        print(f"{'='*60}")
        print(f"  Correlation(V_pred, actual_return): {corr:.3f}")
        print(f"  MAE(V_pred, actual_return):         {mae:.3f}")
        print(f"  Landing rate:                       "
              f"{landed_mask.sum()}/{len(landed_mask)} "
              f"({100*landed_mask.mean():.0f}%)")
        if landed_mask.any():
            print(f"  V_pred  (landed):  mean={v_preds[landed_mask].mean():.2f}  "
                  f"std={v_preds[landed_mask].std():.2f}")
            print(f"  Actual  (landed):  mean={actual_returns[landed_mask].mean():.2f}  "
                  f"std={actual_returns[landed_mask].std():.2f}")
        if (~landed_mask).any():
            print(f"  V_pred  (failed):  mean={v_preds[~landed_mask].mean():.2f}  "
                  f"std={v_preds[~landed_mask].std():.2f}")
            print(f"  Actual  (failed):  mean={actual_returns[~landed_mask].mean():.2f}  "
                  f"std={actual_returns[~landed_mask].std():.2f}")

        # Per-episode detail for first 10
        print(f"\nFirst 10 episodes:")
        print(f"  {'ep':>3}  {'V_pred':>8}  {'actual':>8}  {'outcome':>8}")
        for i in range(min(10, len(v_preds))):
            print(f"  {i:3d}  {v_preds[i]:8.2f}  {actual_returns[i]:8.2f}  {outcomes[i]:>8}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
