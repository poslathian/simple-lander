"""Monitor DAgger training progress.

Checks: latest checkpoint loss, holdout eval, outcome conditioning comparison.
Designed to be run periodically via cron.
"""

import glob
import math
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from model import DiffusionMLP, CosineSchedule, DDIMSampler
from model import X_DIM, COND_DIM, STATE_DIM, CFG_START, CFG_END, N_CPS, N_CHANNELS
from diffusion_controller import (
    KTODiffusionController, Outcome, NORM_SCALES,
    N_CPS, N_CHANNELS, DT,
)
from lunar_lander import LunarLander, KTOController


def load_model(ckpt_path, hidden=1024, n_blocks=6):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model = DiffusionMLP(hidden=hidden, n_blocks=n_blocks)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32)
    schedule = CosineSchedule(T=ckpt.get("T", 100))
    sampler = DDIMSampler(model, schedule, n_steps=10)
    return model, sampler, x_mean, x_std, ckpt


class _Model:
    def __init__(self, sampler, x_mean, x_std, outcome_override=None):
        self.sampler = sampler
        self.x_mean = x_mean
        self.x_std = x_std
        self._outcome_override = outcome_override

    def predict(self, cond, outcome, guidance_scale=2.0):
        actual_outcome = self._outcome_override if self._outcome_override is not None else outcome
        full = np.zeros(COND_DIM, dtype=np.float32)
        full[:STATE_DIM] = cond[:STATE_DIM]
        full[CFG_START] = float(actual_outcome)
        ct = torch.tensor(full, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            x_norm = self.sampler.sample(ct)
        x_raw = (x_norm * self.x_std + self.x_mean).squeeze(0).numpy()
        cps = x_raw.reshape(N_CPS, N_CHANNELS)
        cps[0] = [0, 0, 0]
        return cps


def run_episodes(env, model_wrapper, n_seeds, margin, seed_offset=95000):
    lands = 0
    for i in range(n_seeds):
        seed = seed_offset + i
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0
        ctrl = KTODiffusionController(
            env, model=model_wrapper, target_frequency=3.0,
            action_horizon=1.5, outcome=Outcome.SUCCESS,
        )
        ctrl.warm_start(time_budget=5.0)
        done, step, last_inf = False, 0, -999
        spi = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))
        while not done:
            if step - last_inf >= spi:
                ctrl.inference()
                last_inf = step
            tv, th = ctrl.get_action(guidance_margin=margin)
            obs, r, term, trunc, _ = env.step(np.array([tv, th], dtype=np.float32))
            done = term or trunc
            step += 1
        landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
        lands += landed
    return lands, n_seeds


def monitor(run_dir, hidden=1024, n_blocks=6, n_eval_seeds=20):
    # Find latest checkpoint
    ckpts = sorted(glob.glob(os.path.join(run_dir, "dagger_round*.pt")))
    if not ckpts:
        print(f"No checkpoints in {run_dir}")
        return

    latest = ckpts[-1]
    round_num = int(latest.split("round")[1].split(".")[0])
    print(f"{'='*60}")
    print(f"TRAINING MONITOR — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Run dir: {run_dir}")
    print(f"Latest checkpoint: {latest} (round {round_num})")

    model, sampler, x_mean, x_std, ckpt = load_model(latest, hidden, n_blocks)
    print(f"Epochs: {ckpt.get('epochs', '?')}, Margin: {ckpt.get('margin_mean', '?')}")

    # Register env
    try:
        gym.register(id="LL-monitor", entry_point="lunar_lander:LunarLander",
                     max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-monitor", render_mode=None, continuous=True)

    # Eval at key margins
    print(f"\n--- Margin sweep ({n_eval_seeds} seeds) ---")
    for margin in [0.001, 0.2, 0.5, 0.8, 1.0]:
        wrapper = _Model(sampler, x_mean, x_std)
        lands, total = run_episodes(env, wrapper, n_eval_seeds, margin)
        print(f"  margin={margin:.3f}: {lands}/{total} = {lands/total:.0%}")

    # Outcome conditioning comparison at margin=0.5
    print(f"\n--- Outcome conditioning @ margin=0.5 ({n_eval_seeds} seeds) ---")
    for oc_val, label in [(1, "outcome=+1"), (-1, "outcome=-1")]:
        wrapper = _Model(sampler, x_mean, x_std, outcome_override=oc_val)
        lands, total = run_episodes(env, wrapper, n_eval_seeds, margin=0.5)
        print(f"  {label}: {lands}/{total} = {lands/total:.0%}")

    env.close()

    # Check training log for recent loss values
    log_path = os.path.join(run_dir, "training.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        loss_lines = [l.strip() for l in lines if "loss=" in l]
        if loss_lines:
            print(f"\n--- Recent training losses ---")
            for l in loss_lines[-5:]:
                print(f"  {l}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/F_normalized_h1024")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()
    monitor(args.run_dir, args.hidden, args.n_blocks, args.n_seeds)
