"""Step-by-step PPO for lunar lander in native thrust action space.

Standard closed-loop RL: at each timestep (50 Hz), the policy observes
the 8-dim gymnasium observation and outputs 2-dim continuous actions
[main_engine, side_thruster] in [-1, 1].

Same environment setup (warmup + randomize), reward function, and
500-seed evaluation protocol as the B-spline planner experiments.

Usage:
    python ppo_native.py train --total-steps 2000000
    python ppo_native.py evaluate --ckpt checkpoints_native/best.pt
    python ppo_native.py compare --ckpt checkpoints_native/best.pt --episodes 500
    python ppo_native.py run --ckpt checkpoints_native/best.pt --render
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from kto_lander import (
    DT,
    GRAVITY,
    obs_to_state,
    warmup_and_snapshot,
)


# ===========================================================================
# Constants
# ===========================================================================

OBS_DIM = 8           # gymnasium observation (no leg contacts used as input)
ACTION_DIM = 2        # [main_engine, side_thruster]
EVAL_SEEDS = list(range(5000, 5500))
MAX_EPISODE_STEPS = 500  # 10 seconds at 50 Hz


class Outcome(Enum):
    IN_PROGRESS = 0
    LANDED = 1
    FAILED = 2


def reward_fn(outcome: Outcome, t_elapsed: float) -> float:
    if outcome == Outcome.IN_PROGRESS:
        return 0.0
    elif outcome == Outcome.LANDED:
        return -t_elapsed
    else:
        return -t_elapsed - 10.0


# ===========================================================================
# Policy and Value networks
# ===========================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM,
                 hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            ])
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # Initialize mean head near zero for stable start
        nn.init.uniform_(self.mean_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean), torch.zeros(obs.shape[0], device=obs.device)
        dist = Normal(mean, std)
        x = dist.rsample()
        log_prob = dist.log_prob(x).sum(dim=-1)
        action = torch.tanh(x)
        # Tanh squashing correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob

    def log_prob_of(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(obs)
        # Inverse tanh (atanh)
        x = torch.atanh(action.clamp(-0.999, 0.999))
        dist = Normal(mean, std)
        log_prob = dist.log_prob(x).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return log_prob

    def entropy_approx(self, obs: torch.Tensor) -> torch.Tensor:
        _, std = self.forward(obs)
        return Normal(torch.zeros_like(std), std).entropy().sum(dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 256,
                 num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ===========================================================================
# Rollout buffer
# ===========================================================================

class RolloutBuffer:
    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.episode_returns: list[float] = []
        self.episode_landed: list[bool] = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def finish_episode(self, ep_return: float, landed: bool):
        self.episode_returns.append(ep_return)
        self.episode_landed.append(landed)

    def compute_gae(self, gamma: float = 1.0, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_value = 0.0

        for t in reversed(range(n)):
            if self.dones[t]:
                last_gae = 0.0
                last_value = 0.0

            delta = self.rewards[t] + gamma * last_value * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]
            last_value = self.values[t]

        return torch.tensor(advantages), torch.tensor(returns)

    def to_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.tensor(np.array(self.obs), dtype=torch.float32),
            "actions": torch.tensor(np.array(self.actions), dtype=torch.float32),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
        }

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ===========================================================================
# Episode rollout
# ===========================================================================

def run_episode(policy: PolicyNetwork, value_net: ValueNetwork,
                seed: int, buffer: RolloutBuffer | None = None,
                deterministic: bool = False,
                render: bool = False) -> dict:
    """Run one episode with warmup + randomize, using per-step actions."""
    env = gym.make("LunarLander-v3", continuous=True,
                   render_mode="human" if render else None)
    env.reset(seed=seed)

    try:
        obs_raw, state0, params = warmup_and_snapshot(env, n_steps=5)
    except RuntimeError:
        env.close()
        return {"valid": False, "landed": False, "return": -20.0, "t_elapsed": 0.0}

    obs = obs_raw[:OBS_DIM].copy()
    t = 0.0
    landed = False
    ep_steps = 0

    for step in range(MAX_EPISODE_STEPS):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, log_prob = policy.act(obs_tensor, deterministic=deterministic)
            v = value_net(obs_tensor).item() if buffer is not None else 0.0

        action_np = action.squeeze(0).cpu().numpy()
        log_prob_val = log_prob.item()

        obs_next, _, terminated, truncated, _ = env.step(action_np)
        t += DT
        ep_steps += 1

        # Check landing
        if obs_next[6] > 0.5 and obs_next[7] > 0.5:
            landed = True

        done = terminated or truncated or landed

        # Reward: 0 for non-terminal, sparse terminal reward
        if done:
            outcome = Outcome.LANDED if landed else Outcome.FAILED
            step_reward = reward_fn(outcome, t)
        else:
            step_reward = 0.0

        if buffer is not None:
            buffer.add(obs, action_np, log_prob_val, step_reward, v, done)

        if render:
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN
                        and event.key in (pygame.K_q, pygame.K_ESCAPE)
                    ):
                        env.close()
                        return {"valid": True, "landed": landed, "return": reward_fn(
                            Outcome.LANDED if landed else Outcome.FAILED, t),
                            "t_elapsed": t, "steps": ep_steps, "window_closed": True}
            except Exception:
                pass

        if done:
            break

        obs = obs_next[:OBS_DIM].copy()

    env.close()

    outcome = Outcome.LANDED if landed else Outcome.FAILED
    ep_return = reward_fn(outcome, t)

    if buffer is not None:
        buffer.finish_episode(ep_return, landed)

    return {
        "valid": True, "landed": landed, "return": ep_return,
        "t_elapsed": t, "steps": ep_steps, "window_closed": False,
    }


# ===========================================================================
# PPO Training
# ===========================================================================

@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    steps_per_rollout: int = 4096
    epochs_per_update: int = 10
    minibatch_size: int = 256
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    entropy_coeff: float = 0.01
    gamma: float = 1.0
    gae_lambda: float = 0.95
    min_log_std: float = -3.0
    # Evaluation
    eval_interval: int = 20000
    eval_seeds: int = 100
    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints_native"))


class PPOTrainer:
    def __init__(self, policy: PolicyNetwork, value: ValueNetwork,
                 config: PPOConfig | None = None):
        self.policy = policy
        self.value = value
        self.config = config or PPOConfig()
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=self.config.lr_policy)
        self.value_optimizer = torch.optim.Adam(
            value.parameters(), lr=self.config.lr_value)

    def train(self) -> dict:
        cfg = self.config
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_landing = 0.0
        best_return = float("-inf")
        total_steps = 0
        total_episodes = 0
        seed_counter = 50000  # training seeds far from eval seeds
        update_count = 0
        next_eval_at = cfg.eval_interval

        print(f"[ppo-native] starting training: {cfg.total_steps} total steps, "
              f"{cfg.steps_per_rollout} steps/rollout")

        t_start = time.perf_counter()
        last_report = t_start

        while total_steps < cfg.total_steps:
            # Collect rollout
            buffer = RolloutBuffer()

            while len(buffer) < cfg.steps_per_rollout:
                seed_counter += 1
                result = run_episode(
                    self.policy, self.value, seed=seed_counter, buffer=buffer)

            total_steps += len(buffer)
            total_episodes += len(buffer.episode_returns)

            # Compute GAE
            advantages, returns = buffer.compute_gae(cfg.gamma, cfg.gae_lambda)
            tensors = buffer.to_tensors()
            obs = tensors["obs"]
            actions = tensors["actions"]
            old_log_probs = tensors["log_probs"]

            # Normalize advantages
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            n = len(buffer)
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            entropy_sum = 0.0
            n_updates = 0

            for epoch in range(cfg.epochs_per_update):
                perm = torch.randperm(n)
                for i in range(0, n, cfg.minibatch_size):
                    idx = perm[i:i+cfg.minibatch_size]

                    mb_obs = obs[idx]
                    mb_actions = actions[idx]
                    mb_old_log_probs = old_log_probs[idx]
                    mb_advantages = advantages[idx]
                    mb_returns = returns[idx]

                    # Policy loss
                    new_log_probs = self.policy.log_prob_of(mb_obs, mb_actions)
                    ratio = (new_log_probs - mb_old_log_probs).exp()
                    clipped = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                    policy_loss = -torch.min(
                        ratio * mb_advantages, clipped * mb_advantages).mean()

                    entropy = self.policy.entropy_approx(mb_obs).mean()
                    loss_policy = policy_loss - cfg.entropy_coeff * entropy

                    self.policy_optimizer.zero_grad()
                    loss_policy.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm)
                    self.policy_optimizer.step()

                    with torch.no_grad():
                        self.policy.log_std.clamp_(min=cfg.min_log_std)

                    # Value loss
                    v_pred = self.value(mb_obs)
                    value_loss = F.mse_loss(v_pred, mb_returns)

                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.value.parameters(), cfg.max_grad_norm)
                    self.value_optimizer.step()

                    policy_loss_sum += policy_loss.item()
                    value_loss_sum += value_loss.item()
                    entropy_sum += entropy.item()
                    n_updates += 1

            update_count += 1

            # Log
            ep_rets = buffer.episode_returns
            ep_lands = buffer.episode_landed
            landing_rate = sum(ep_lands) / max(len(ep_lands), 1)
            mean_ret = np.mean(ep_rets) if ep_rets else 0.0
            log_std_mean = self.policy.log_std.data.mean().item()

            elapsed = time.perf_counter() - t_start
            sps = total_steps / max(elapsed, 1)
            print(f"  steps={total_steps:>8d}  eps={total_episodes:>5d}  "
                  f"land={100*landing_rate:.0f}%  ret={mean_ret:.2f}  "
                  f"ploss={policy_loss_sum/n_updates:.4f}  "
                  f"vloss={value_loss_sum/n_updates:.4f}  "
                  f"ent={entropy_sum/n_updates:.3f}  "
                  f"logstd={log_std_mean:.2f}  "
                  f"[{sps:.0f} sps]", flush=True)

            # Periodic report
            now = time.perf_counter()
            if now - last_report >= 900:  # 15 min
                last_report = now
                self._inject_progress(total_steps, landing_rate, mean_ret)

            # Evaluate
            if total_steps >= next_eval_at:
                next_eval_at += cfg.eval_interval
                eval_metrics = self._evaluate(cfg.eval_seeds)
                print(f"  EVAL({cfg.eval_seeds} seeds): "
                      f"land={100*eval_metrics['landing_rate']:.0f}%  "
                      f"ret={eval_metrics['mean_return']:.2f}", flush=True)

                improved = (eval_metrics["landing_rate"] > best_landing or
                            (eval_metrics["landing_rate"] == best_landing and
                             eval_metrics["mean_return"] > best_return))
                if improved:
                    best_landing = eval_metrics["landing_rate"]
                    best_return = eval_metrics["mean_return"]
                    self._save_checkpoint(cfg.checkpoint_dir / "best.pt")
                    print(f"  NEW BEST! saved checkpoint", flush=True)

            buffer.clear()

        self._save_checkpoint(cfg.checkpoint_dir / "last.pt")
        return {
            "best_landing_rate": best_landing,
            "best_mean_return": best_return,
            "total_steps": total_steps,
            "total_episodes": total_episodes,
        }

    def _inject_progress(self, steps, landing_rate, mean_ret):
        import subprocess
        msg = (f"simple-rl: native PPO progress — {steps/1e6:.1f}M steps, "
               f"train land={100*landing_rate:.0f}%, ret={mean_ret:.2f}")
        try:
            subprocess.run(
                [str(Path.home() / "habitat3/hooks/inject.sh"), "hab3-ctrl", msg],
                capture_output=True, timeout=5)
        except Exception:
            pass

    def _evaluate(self, n_seeds: int, seeds: list[int] | None = None) -> dict:
        if seeds is None:
            seeds = EVAL_SEEDS[:n_seeds]
        landed = 0
        total_return = 0.0
        valid = 0

        self.policy.eval()
        for seed in seeds:
            result = run_episode(
                self.policy, self.value, seed=seed, deterministic=True)
            if not result["valid"]:
                continue
            valid += 1
            if result["landed"]:
                landed += 1
            total_return += result["return"]

        self.policy.train()
        n = max(valid, 1)
        return {
            "landing_rate": landed / n,
            "mean_return": total_return / n,
            "valid": valid,
        }

    def _save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }, path)

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])
        if "policy_optimizer" in ckpt:
            self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        if "value_optimizer" in ckpt:
            self.value_optimizer.load_state_dict(ckpt["value_optimizer"])


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Native action-space PPO for lunar lander")
    sub = parser.add_subparsers(dest="cmd")

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--total-steps", type=int, default=2_000_000)
    p_train.add_argument("--lr-policy", type=float, default=3e-4)
    p_train.add_argument("--lr-value", type=float, default=1e-3)
    p_train.add_argument("--clip-eps", type=float, default=0.2)
    p_train.add_argument("--eval-interval", type=int, default=20000)
    p_train.add_argument("--eval-seeds", type=int, default=100)
    p_train.add_argument("--resume", type=str, default=None)

    # evaluate
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--ckpt", type=str, default="checkpoints_native/best.pt")
    p_eval.add_argument("--seeds", type=int, default=500)

    # run
    p_run = sub.add_parser("run")
    p_run.add_argument("--ckpt", type=str, default="checkpoints_native/best.pt")
    p_run.add_argument("--render", action="store_true")
    p_run.add_argument("--seed", type=int, default=0)

    # compare
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--ckpt", type=str, default="checkpoints_native/best.pt")
    p_cmp.add_argument("--episodes", type=int, default=500)

    args = parser.parse_args()

    if args.cmd == "train":
        policy = PolicyNetwork()
        value = ValueNetwork()
        n_policy = sum(p.numel() for p in policy.parameters())
        n_value = sum(p.numel() for p in value.parameters())
        print(f"[ppo-native] policy: {n_policy:,} params, value: {n_value:,} params")

        cfg = PPOConfig(
            total_steps=args.total_steps,
            lr_policy=args.lr_policy,
            lr_value=args.lr_value,
            clip_eps=args.clip_eps,
            eval_interval=args.eval_interval,
            eval_seeds=args.eval_seeds,
        )

        trainer = PPOTrainer(policy, value, cfg)
        if args.resume:
            trainer._load_checkpoint(Path(args.resume))
            print(f"[ppo-native] resumed from {args.resume}")

        result = trainer.train()
        print(f"\n[ppo-native] DONE — best landing={100*result['best_landing_rate']:.0f}% "
              f"return={result['best_mean_return']:.2f}")

    elif args.cmd == "evaluate":
        ckpt = torch.load(Path(args.ckpt), weights_only=False)
        policy = PolicyNetwork()
        policy.load_state_dict(ckpt["policy"])
        value = ValueNetwork()
        value.load_state_dict(ckpt["value"])

        trainer = PPOTrainer(policy, value)
        seeds = EVAL_SEEDS[:args.seeds]
        metrics = trainer._evaluate(len(seeds), seeds)

        print(f"\n{'='*60}")
        print(f"NATIVE PPO EVALUATION over {args.seeds} seeds")
        print(f"{'='*60}")
        print(f"  Landing rate:  {metrics['landing_rate']:.1%} "
              f"({int(metrics['landing_rate'] * metrics['valid'])}/{metrics['valid']})")
        print(f"  Mean return:   {metrics['mean_return']:.2f}")

    elif args.cmd == "run":
        ckpt = torch.load(Path(args.ckpt), weights_only=False)
        policy = PolicyNetwork()
        policy.load_state_dict(ckpt["policy"])
        value = ValueNetwork()
        value.load_state_dict(ckpt["value"])

        if args.render:
            seed = args.seed
            while True:
                result = run_episode(policy, value, seed=seed,
                                     deterministic=True, render=True)
                if result.get("window_closed"):
                    break
                status = "LANDED" if result["landed"] else "FAILED"
                print(f"[seed {seed}] {status} t={result['t_elapsed']:.2f}s "
                      f"return={result['return']:.2f}")
                seed += 1
        else:
            result = run_episode(policy, value, seed=args.seed, deterministic=True)
            status = "LANDED" if result["landed"] else "FAILED"
            print(f"[ppo-native] {status} t={result['t_elapsed']:.2f}s "
                  f"return={result['return']:.2f}")

    elif args.cmd == "compare":
        from kto_lander import plan_with_kto, TrackerGains, control
        from ppo_lander import encoding_to_plan, rollout_plan

        ckpt = torch.load(Path(args.ckpt), weights_only=False)
        policy = PolicyNetwork()
        policy.load_state_dict(ckpt["policy"])
        value = ValueNetwork()
        value.load_state_dict(ckpt["value"])

        seeds = EVAL_SEEDS[:args.episodes]
        kto_results = []
        native_results = []

        print(f"Running {args.episodes} episodes each for KTO and native PPO...")
        for i, seed in enumerate(seeds):
            # KTO
            env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
            env.reset(seed=seed)
            try:
                obs_raw, state0, params = warmup_and_snapshot(env, n_steps=5)
                x0, y0, vx0, vy0 = state0[0], state0[1], state0[2], state0[3]
                plan, plan_T, _ = plan_with_kto(
                    x0, y0, vx0, vy0, params, verbose=False)
                cps = []
                for cp in plan.traj.control_points():
                    arr = np.asarray(cp).flatten()
                    cps.extend(arr[:2].tolist())
                plan_enc = np.concatenate([cps, [plan_T]])
                kto_r = rollout_plan(plan_enc, env, obs_raw, state0, params)
            except (RuntimeError, Exception):
                kto_r = {"landed": False, "return": -20.0, "planning_failed": True}
            env.close()
            kto_results.append(kto_r)

            # Native PPO
            native_r = run_episode(policy, value, seed=seed, deterministic=True)
            native_results.append(native_r)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{args.episodes}...", flush=True)

        kto_valid = [r for r in kto_results if not r.get("planning_failed")]
        native_valid = [r for r in native_results if r.get("valid", True)]
        kto_landed = sum(1 for r in kto_valid if r["landed"])
        native_landed = sum(1 for r in native_valid if r["landed"])
        kto_returns = [r["return"] for r in kto_valid]
        native_returns = [r["return"] for r in native_valid]

        print(f"\n{'='*60}")
        print(f"COMPARISON: KTO vs Native PPO ({args.episodes} seeds)")
        print(f"{'='*60}")
        print(f"  {'Metric':<25} {'KTO':>10} {'NativePPO':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10}")
        print(f"  {'Landing rate':<25} "
              f"{kto_landed}/{len(kto_valid):>5} "
              f"{native_landed}/{len(native_valid):>5}")
        print(f"  {'Landing %':<25} "
              f"{100*kto_landed/max(len(kto_valid),1):>9.0f}% "
              f"{100*native_landed/max(len(native_valid),1):>9.0f}%")
        if kto_returns and native_returns:
            print(f"  {'Mean return':<25} "
                  f"{np.mean(kto_returns):>10.2f} "
                  f"{np.mean(native_returns):>10.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
