"""Evaluate diffusion model: landing rates, diffusion MSE, and training loss.

Reports per-margin landing rates, the diffusion model's MSE on held-out
training frames, and the final training loss from the checkpoint.
"""

import numpy as np
import gymnasium as gym
import torch

from lunar_lander import LunarLander, KTOController, DT, TIMEOUT
from diffusion_controller import (
    DiffusionController, LanderState, ActionTarget,
    WaypointResult,
)


def obs_to_lander_state(obs, action=(0.0, 0.0)):
    return LanderState(
        t_sim_lander=float(obs[8]),
        q=(float(obs[0]), float(obs[1]), float(obs[4])),
        q_prime=(float(obs[2]), float(obs[3]), float(obs[5])),
        thrust=(float(action[0]), float(action[1])),
        contacts=(bool(obs[6]), bool(obs[7]), False),
    )


def rollout_with_margin(env, seed, margin):
    """Run one episode with KTO guidance at the given margin."""
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    uw.lander.linearVelocity = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0

    kto_ctrl = KTOController(env)
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


def compute_diffusion_mse(model, schedule, norm_stats, db_path="rollouts.db", max_frames=500):
    """Compute MSE of the diffusion model on held-out training frames.

    Runs one-step denoising from t=1 (minimal noise) and compares
    predicted x0 against ground truth CPs.
    """
    from rollout_db import RolloutDB
    from model import STATE_DIM, CFG_START, CFG_END

    db = RolloutDB(db_path)
    conds, targets = [], []
    n = db.count()
    for ep_id in range(1, n + 1):
        for cond, target_cps, cfg_wp, cfg_crash_t in db.get_training_frames(ep_id):
            conds.append(cond)
            targets.append(target_cps)
            if len(conds) >= max_frames:
                break
        if len(conds) >= max_frames:
            break
    db.close()

    if not conds:
        return float("nan")

    conds = torch.tensor(np.array(conds), dtype=torch.float32)
    targets = torch.tensor(np.array(targets), dtype=torch.float32)

    x_mean = torch.tensor(norm_stats["x_mean"])
    x_std = torch.tensor(norm_stats["x_std"])
    x0_norm = (targets - x_mean) / x_std

    # Light noise at t=1
    alpha_bar = schedule.get_alpha_bar(1)
    noise = torch.randn_like(x0_norm)
    x_noisy = np.sqrt(alpha_bar) * x0_norm + np.sqrt(1 - alpha_bar) * noise

    with torch.no_grad():
        t_batch = torch.ones(len(conds), dtype=torch.long)
        eps_pred = model(x_noisy, conds, t_batch)

    # Reconstruct x0 from noise prediction
    x0_pred = (x_noisy - np.sqrt(1 - alpha_bar) * eps_pred) / np.sqrt(alpha_bar)
    x0_pred_raw = x0_pred * x_std + x_mean

    mse = float(torch.mean((x0_pred_raw - targets) ** 2))
    return mse


def evaluate(model_path=None, n_baseline=100, n_per_margin=10, seed_offset=1000,
             db_path="rollouts.db"):
    """Run full evaluation: MSE, loss, and landing rates per margin."""
    gym.register(id="LL-eval", entry_point="lunar_lander:LunarLander",
                 max_episode_steps=1000)
    env = gym.make("LL-eval", render_mode=None, continuous=True)

    diff_mse = float("nan")
    train_loss = float("nan")
    train_epochs = 0
    train_frames = 0

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

        train_epochs = checkpoint.get("epochs", 0)
        train_frames = checkpoint.get("n_frames", 0)
        train_loss = checkpoint.get("final_loss", float("nan"))

        # Compute diffusion MSE
        import os
        if os.path.exists(db_path):
            diff_mse = compute_diffusion_mse(model, schedule, norm_stats, db_path)

        print(f"Model: {model_path}")
        print(f"  Training: {train_epochs} epochs, {train_frames} frames, loss={train_loss:.4f}")
        print(f"  Diffusion MSE: {diff_mse:.4f}")

    results = {}

    # Baseline: tight guidance (0.001)
    print(f"\nBaseline (margin=0.001, {n_baseline} episodes)...")
    rewards, lands = [], []
    for i in range(n_baseline):
        r, landed = rollout_with_margin(env, seed_offset + i, 0.001)
        rewards.append(r)
        lands.append(landed)
    results["baseline"] = {
        "margin": 0.001,
        "n": n_baseline,
        "land_rate": np.mean(lands),
        "mean_reward": np.mean(rewards),
    }
    print(f"  land_rate={np.mean(lands):.0%} ({sum(lands)}/{n_baseline})  reward={np.mean(rewards):.2f}")

    # Sweep margins
    margins = [0.1 * i for i in range(1, 11)]
    for margin in margins:
        print(f"\nMargin={margin:.1f} ({n_per_margin} episodes)...")
        rewards, lands = [], []
        for i in range(n_per_margin):
            r, landed = rollout_with_margin(env, seed_offset + i, margin)
            rewards.append(r)
            lands.append(landed)
        results[f"margin_{margin:.1f}"] = {
            "margin": margin,
            "n": n_per_margin,
            "land_rate": np.mean(lands),
            "mean_reward": np.mean(rewards),
        }
        print(f"  land_rate={np.mean(lands):.0%} ({sum(lands)}/{n_per_margin})  reward={np.mean(rewards):.2f}")

    env.close()

    # Summary table
    print(f"\n{'Margin':>8}  {'N':>4}  {'Land':>6}  {'Land%':>6}  {'Reward':>8}")
    print("-" * 40)
    for key, v in results.items():
        landed_n = int(v['land_rate'] * v['n'])
        print(f"{v['margin']:8.3f}  {v['n']:4d}  {landed_n:3d}/{v['n']:<3d} {v['land_rate']:5.0%}  {v['mean_reward']:8.2f}")

    if model_path:
        print(f"\nDiffusion MSE: {diff_mse:.4f}")
        print(f"Train loss: {train_loss:.4f} ({train_epochs} epochs, {train_frames} frames)")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to trained model.pt")
    parser.add_argument("--db", default="rollouts.db", help="DB for MSE computation")
    parser.add_argument("--n-baseline", type=int, default=100)
    parser.add_argument("--n-per-margin", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=1000)
    args = parser.parse_args()

    evaluate(args.model, args.n_baseline, args.n_per_margin, args.seed_offset, args.db)
