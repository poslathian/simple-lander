"""Evaluate diffusion model across guidance margin strengths.

Runs 100 random seeds as baseline (guidance margin 0.001), then
10 rollouts each at margins [0.1, 0.2, ..., 1.0] to measure how
the model performs as guidance loosens.
"""

import numpy as np
import gymnasium as gym
import torch

from lunar_lander import LunarLander, KTOController, DT, TIMEOUT
from diffusion_controller import (
    DiffusionController, LanderState, ActionTarget, _build_cond,
    WaypointTarget, WaypointResult, _obs_to_world,
)
from guidance_controller import GuidanceController, GUIDANCE_MARGIN
from model import PAD_CX, PAD_Y


def obs_to_lander_state(obs, action=(0.0, 0.0)):
    return LanderState(
        t_sim_lander=float(obs[8]),
        q=(float(obs[0]), float(obs[1]), float(obs[4])),
        q_prime=(float(obs[2]), float(obs[3]), float(obs[5])),
        thrust=(float(action[0]), float(action[1])),
        contacts=(bool(obs[6]), bool(obs[7]), False),
    )


def rollout_with_margin(env, seed, margin, model_path=None):
    """Run one episode with KTO guidance at the given margin.

    If model_path is provided, loads trained weights. Otherwise uses
    random weights (the post-inference clamping still applies).
    """
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    uw.lander.linearVelocity = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0

    kto_ctrl = KTOController(env, time_budget=5.0)
    total_reward, done = 0.0, False
    prev_action = np.array([0.0, 0.0], dtype=np.float32)

    while not done:
        action = kto_ctrl.step(env)

        at = ActionTarget(
            thrust_v=float(action[0]), thrust_v_margin=margin,
            thrust_h=float(action[1]), thrust_h_margin=margin,
            thrust_t=DT, thrust_t_margin=0.001,
        )
        state = obs_to_lander_state(obs, prev_action)
        spline = DiffusionController(
            timeout=TIMEOUT,
            t_obs_cmd_latency=DT,
            obstacles=[],
            lander_state=state,
            waypoint_goals=[],
            guidance_actions=[at],
            classifier_free_guidance=[WaypointResult(outcome=1, alpha=1.0)],
            action_horizon=0.1,
            target_frequency=50.0,
        )
        tv, th = spline(DT)
        action_out = np.array([tv, th], dtype=np.float32)

        obs, reward, term, trunc, _ = env.step(action_out)
        total_reward += reward
        prev_action = action_out
        done = term or trunc

    landed = (
        not uw.game_over
        and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
    )
    return total_reward, landed


def evaluate(model_path=None, n_baseline=100, n_per_margin=10, seed_offset=1000):
    """Run evaluation across margin strengths."""
    gym.register(id="LL-eval", entry_point="lunar_lander:LunarLander",
                 max_episode_steps=1000)
    env = gym.make("LL-eval", render_mode=None, continuous=True)

    # Load trained model if provided
    if model_path:
        from model import DiffusionMLP, CosineSchedule, DDIMSampler, X_DIM
        from diffusion_controller import _MODEL_CACHE
        checkpoint = torch.load(model_path, weights_only=False)
        model = DiffusionMLP()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        schedule = CosineSchedule(T=100)
        sampler = DDIMSampler(model, schedule, n_steps=10)
        norm_stats = {
            "x_mean": checkpoint["x_mean"],
            "x_std": checkpoint["x_std"],
        }
        _MODEL_CACHE["sampler"] = sampler
        _MODEL_CACHE["norm_stats"] = norm_stats
        print(f"Loaded model from {model_path}")

    results = {}

    # Baseline: tight guidance (0.001)
    print(f"\nBaseline (margin=0.001, {n_baseline} episodes)...")
    rewards, lands = [], []
    for i in range(n_baseline):
        r, landed = rollout_with_margin(env, seed_offset + i, 0.001, model_path)
        rewards.append(r)
        lands.append(landed)
    results["baseline"] = {
        "margin": 0.001,
        "n": n_baseline,
        "land_rate": np.mean(lands),
        "mean_reward": np.mean(rewards),
    }
    print(f"  land_rate={np.mean(lands):.0%}  reward={np.mean(rewards):.2f}")

    # Sweep margins
    margins = [0.1 * i for i in range(1, 11)]
    for margin in margins:
        print(f"\nMargin={margin:.1f} ({n_per_margin} episodes)...")
        rewards, lands = [], []
        for i in range(n_per_margin):
            r, landed = rollout_with_margin(env, seed_offset + i, margin, model_path)
            rewards.append(r)
            lands.append(landed)
        results[f"margin_{margin:.1f}"] = {
            "margin": margin,
            "n": n_per_margin,
            "land_rate": np.mean(lands),
            "mean_reward": np.mean(rewards),
        }
        print(f"  land_rate={np.mean(lands):.0%}  reward={np.mean(rewards):.2f}")

    env.close()

    # Summary table
    print(f"\n{'Margin':>8}  {'N':>4}  {'Land%':>6}  {'Reward':>8}")
    print("-" * 32)
    for key, v in results.items():
        print(f"{v['margin']:8.3f}  {v['n']:4d}  {v['land_rate']:5.0%}  {v['mean_reward']:8.2f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to trained model.pt")
    parser.add_argument("--n-baseline", type=int, default=100)
    parser.add_argument("--n-per-margin", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=1000)
    args = parser.parse_args()

    evaluate(args.model, args.n_baseline, args.n_per_margin, args.seed_offset)
