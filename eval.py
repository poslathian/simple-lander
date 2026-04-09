"""Evaluate trained diffusion model across guidance margin strengths.

Runs KTO baseline first, then sweeps margins [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0].
Success = any margin with more landings than 0.001 baseline.
"""

import sys
import time

import gymnasium as gym
import numpy as np
import torch

from lunar_lander import LunarLander, KTOController, DT, TIMEOUT
from diffusion_controller import (
    KTODiffusionController, NoiseModel, Outcome,
    N_CPS, N_CHANNELS, COND_DIM, STATE_DIM,
    _build_cond, ObstacleRelative, WaypointTarget,
)
from model import DiffusionMLP, CosineSchedule, DDIMSampler, CFG_START, CFG_END
import solver


# ── Trained model wrapper ─────────────────────────────────────────────────

class TrainedModel:
    """Wraps trained DiffusionMLP for use with KTODiffusionController."""

    def __init__(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model = DiffusionMLP()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        T = checkpoint.get("T", 100)
        schedule = CosineSchedule(T=T)
        self.sampler = DDIMSampler(self.model, schedule, n_steps=10)

        self.x_mean = torch.tensor(checkpoint["x_mean"], dtype=torch.float32)
        self.x_std = torch.tensor(checkpoint["x_std"], dtype=torch.float32)

    def predict(self, cond, outcome, guidance_scale=2.0):
        # Build full 21-dim conditioning
        full_cond = np.zeros(COND_DIM, dtype=np.float32)
        full_cond[:STATE_DIM] = cond[:STATE_DIM]
        full_cond[CFG_START] = float(outcome)  # +1 for success

        cond_tensor = torch.tensor(full_cond, dtype=torch.float32).unsqueeze(0)
        x_norm = self.sampler.sample(cond_tensor)

        # Denormalize
        x_raw = (x_norm * self.x_std + self.x_mean).squeeze(0).numpy()

        # Reshape to (10, 3)
        cps = x_raw.reshape(N_CPS, N_CHANNELS)
        cps[0] = [0.0, 0.0, 0.0]  # Pin first CP to origin
        return cps


# ── Run episode ───────────────────────────────────────────────────────────

def run_episode(env, seed, model, margin, kto_cache=None, plan_pool=None,
                outcome=Outcome.SUCCESS):
    """Run one episode with KTODiffusionController."""
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    uw.lander.linearVelocity = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0

    ctrl = KTODiffusionController(
        env, model=model, target_frequency=3.0,
        action_horizon=1.5, outcome=outcome,
    )
    cached = plan_pool.get(seed) if plan_pool else None
    if cached is not None:
        ctrl._kto = type("CachedKTO", (), {
            "plan": cached["plan"], "n_steps": cached["n_steps"],
        })()
        ctrl._kto_t0 = uw.elapsed_s
        ctrl._kto_duration = cached["n_steps"] * DT
    else:
        ctrl.warm_start(time_budget=5.0)

    # Cache KTO plan for reuse across margins
    if kto_cache is not None and seed not in kto_cache:
        kto_cache[seed] = {
            "plan": ctrl._kto.plan,
            "n_steps": ctrl._kto.n_steps,
            "t0": ctrl._kto_t0,
        }

    total_reward = 0.0
    done = False
    step_idx = 0
    last_inference_step = -999
    steps_per_inference = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))

    while not done:
        if step_idx - last_inference_step >= steps_per_inference:
            ctrl.inference()
            last_inference_step = step_idx

        tv, th = ctrl.get_action(guidance_margin=margin)
        action_out = np.array([tv, th], dtype=np.float32)
        obs, reward, term, trunc, _ = env.step(action_out)
        total_reward += reward
        done = term or trunc
        step_idx += 1

    landed = (
        not uw.game_over
        and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
    )
    return total_reward, landed


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to position_model.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=8000)
    args = parser.parse_args()

    gym.register(
        id="LL-eval",
        entry_point="lunar_lander:LunarLander",
        max_episode_steps=1000,
    )
    env = gym.make("LL-eval", render_mode=None, continuous=True)
    seeds = [args.seed_offset + i for i in range(args.episodes)]
    kto_cache = {}

    # Load trained model
    trained = TrainedModel(args.model)
    print(f"Loaded model from {args.model}")

    # ── KTO baseline (no diffusion) ──
    print(f"\nRunning KTO baseline ({args.episodes} episodes)...")
    kto_lands = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0
        kto = KTOController(env, time_budget=5.0)
        done = False
        while not done:
            action = kto.step(env)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
        landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
        kto_lands += landed
    kto_rate = kto_lands / args.episodes
    print(f"  KTO baseline: {kto_lands}/{args.episodes} = {kto_rate:.0%}")

    # ── Margin sweep with NoiseModel (0.001 baseline) ──
    print(f"\nRunning NoiseModel baseline (margin=0.001)...")
    noise_model = NoiseModel()
    noise_lands = 0
    for seed in seeds:
        _, landed = run_episode(env, seed, noise_model, 0.001, kto_cache)
        noise_lands += landed
    noise_rate = noise_lands / args.episodes
    print(f"  NoiseModel m=0.001: {noise_lands}/{args.episodes} = {noise_rate:.0%}")

    # ── Trained model at various margins ──
    margins = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    results = {}

    for margin in margins:
        t0 = time.time()
        lands = 0
        for seed in seeds:
            _, landed = run_episode(env, seed, trained, margin, kto_cache)
            lands += landed
        elapsed = time.time() - t0
        rate = lands / args.episodes
        results[margin] = {"lands": lands, "rate": rate}
        print(f"  Trained m={margin:.3f}: {lands}/{args.episodes} = {rate:.0%}  ({elapsed:.1f}s)")

    env.close()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"{'Margin':>8}  {'Lands':>6}  {'Rate':>6}  {'vs baseline':>12}")
    print(f"{'='*60}")
    print(f"{'KTO':>8}  {kto_lands:>6}  {kto_rate:>5.0%}  {'(reference)':>12}")
    print(f"{'noise':>8}  {noise_lands:>6}  {noise_rate:>5.0%}  {'(m=0.001)':>12}")
    for margin, r in results.items():
        delta = r["lands"] - noise_lands
        sign = "+" if delta >= 0 else ""
        print(f"{margin:>8.3f}  {r['lands']:>6}  {r['rate']:>5.0%}  {sign}{delta:>11}")
    print(f"{'='*60}")

    # Check success: any margin beats noise baseline
    best_margin = max(results.keys(), key=lambda m: results[m]["lands"])
    best = results[best_margin]
    if best["lands"] > noise_lands:
        print(f"\nSUCCESS: margin={best_margin} lands {best['lands']} > noise baseline {noise_lands}")
        return 0
    else:
        print(f"\nNo margin beat the noise baseline ({noise_lands} lands)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
