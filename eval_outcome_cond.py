"""Sanity check: does outcome conditioning actually work?

Runs eval at margin=1.0 with outcome_cond set to -1, 0, and +1.
We expect:
  outcome=+1: ~64-70% landing (what we've been training with)
  outcome=0:  fewer landings (unconditional, no success steering)
  outcome=-1: near-zero landings (actively steering toward failure)
"""

import argparse
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from diffusion_controller import (
    KTODiffusionController, Outcome,
    N_CPS, N_CHANNELS, COND_DIM, STATE_DIM, DT,
)
from model import DiffusionMLP, CosineSchedule, DDIMSampler, CFG_START, CFG_END
from lunar_lander import LunarLander, KTOController


class _Model:
    def __init__(self, checkpoint_path, hidden=256, n_blocks=6):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        self.model = DiffusionMLP(hidden=hidden, n_blocks=n_blocks)
        try:
            self.model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()
        schedule = CosineSchedule(T=ckpt.get("T", 100))
        self.sampler = DDIMSampler(self.model, schedule, n_steps=10)
        self.x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
        self.x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32)
        self._outcome_override = None

    def predict(self, cond, outcome, guidance_scale=2.0):
        # Use the override if set, otherwise use the passed outcome
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


def run_episode(env, seed, model, margin):
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    uw.lander.linearVelocity = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0

    ctrl = KTODiffusionController(
        env, model=model, target_frequency=3.0,
        action_horizon=1.5, outcome=Outcome.SUCCESS,
    )
    ctrl.warm_start(time_budget=5.0)

    done = False
    step_idx = 0
    last_inf = -999
    spi = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))

    while not done:
        if step_idx - last_inf >= spi:
            ctrl.inference()
            last_inf = step_idx
        tv, th = ctrl.get_action(guidance_margin=margin)
        obs, r, term, trunc, _ = env.step(np.array([tv, th], dtype=np.float32))
        done = term or trunc
        step_idx += 1

    landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
    return landed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=90000)
    args = parser.parse_args()

    try:
        gym.register(id="LL-oceval", entry_point="lunar_lander:LunarLander",
                     max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-oceval", render_mode=None, continuous=True)

    model = _Model(args.checkpoint, hidden=args.hidden, n_blocks=args.n_blocks)
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))

    print(f"Outcome conditioning sanity check")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  margin: {args.margin}")
    print(f"  seeds: {args.n_seeds} ({args.seed_offset}-{args.seed_offset + args.n_seeds - 1})")
    print(f"{'='*50}")

    for outcome_val, label in [(1, "outcome=+1 (success)"),
                                (0, "outcome= 0 (unconditional)"),
                                (-1, "outcome=-1 (failure)")]:
        model._outcome_override = outcome_val
        t0 = time.time()
        lands = 0
        for seed in seeds:
            landed = run_episode(env, seed, model, args.margin)
            lands += landed
        elapsed = time.time() - t0
        rate = lands / args.n_seeds
        print(f"  {label}: {lands}/{args.n_seeds} = {rate:.0%}  ({elapsed:.0f}s)")

    env.close()
    print(f"{'='*50}")


if __name__ == "__main__":
    sys.exit(main())
