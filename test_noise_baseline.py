"""Step 0 smoke test: KTODiffusionController with NoiseModel reproduces KTO baseline.

Validates the full pipeline: KTO warm-start, noise model inference at 3Hz,
PD tracking with guidance_margin=0.001, zero thrust after KTO exhausted.

Pass criteria:
  - Landing rate within 5% of KTO baseline (same seeds)
  - Mean position tracking RMS < 0.5m
"""

import os
import sys
import time

import gymnasium as gym
import numpy as np
from PIL import Image

from lunar_lander import LunarLander, KTOController, DT, TIMEOUT
from diffusion_controller import KTODiffusionController, NoiseModel, Outcome
import solver


GUIDANCE_MARGIN = 0.001
N_EPISODES = 100
SEED_OFFSET = 2000


def run_episode(env, seed, save_frames=False, frame_dir=None):
    """Run one episode with KTODiffusionController."""
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    uw.lander.linearVelocity = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0

    ctrl = KTODiffusionController(
        env,
        model=NoiseModel(),
        target_frequency=3.0,
        action_horizon=1.5,
        outcome=Outcome.SUCCESS,
    )
    ctrl.warm_start(time_budget=5.0)

    total_reward = 0.0
    done = False
    step_idx = 0
    tracking_errors = []
    frames = []
    last_inference_step = -999

    while not done:
        t_sim = uw.elapsed_s

        # Run inference at target_frequency (every ~17 steps at 50Hz/3Hz)
        steps_per_inference = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))
        if step_idx - last_inference_step >= steps_per_inference:
            ctrl.inference()
            last_inference_step = step_idx

        # Get blended action
        tv, th = ctrl.get_action(guidance_margin=GUIDANCE_MARGIN)
        action_out = np.array([tv, th], dtype=np.float32)

        # Track error vs KTO reference
        kto_ref = ctrl._get_kto_ref(t_sim)
        if kto_ref is not None:
            L = uw.lander
            pos_err = np.sqrt(
                (L.position.x - kto_ref["q"][0])**2 +
                (L.position.y - kto_ref["q"][1])**2
            )
            tracking_errors.append(pos_err)

        # Save frame
        if save_frames and hasattr(env, 'render'):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        obs, reward, term, trunc, _ = env.step(action_out)
        total_reward += reward
        done = term or trunc
        step_idx += 1

    landed = (
        not uw.game_over
        and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
    )

    if save_frames and frames and frame_dir:
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(frame_dir, f"frame_{i:04d}.png"))
        print(f"  Saved {len(frames)} frames to {frame_dir}/")

    rms_err = np.sqrt(np.mean(np.array(tracking_errors)**2)) if tracking_errors else 0.0
    return total_reward, landed, rms_err


def run_kto_baseline(env, seeds):
    """Run pure KTO baseline for comparison."""
    landed_count = 0
    rewards = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0
        kto = KTOController(env, time_budget=5.0)
        done, total_r = False, 0.0
        while not done:
            action = kto.step(env)
            obs, reward, term, trunc, _ = env.step(action)
            total_r += reward
            done = term or trunc
        landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
        landed_count += landed
        rewards.append(total_r)
    return landed_count / len(seeds), np.mean(rewards)


def main():
    gym.register(
        id="LL-test-noise",
        entry_point="lunar_lander:LunarLander",
        max_episode_steps=1000,
    )

    seeds = [SEED_OFFSET + i for i in range(N_EPISODES)]

    # ── KTO baseline ──
    print(f"Running KTO baseline ({N_EPISODES} episodes)...")
    env_kto = gym.make("LL-test-noise", render_mode=None, continuous=True)
    kto_land_rate, kto_mean_reward = run_kto_baseline(env_kto, seeds)
    env_kto.close()
    print(f"  KTO baseline: land_rate={kto_land_rate:.0%}  reward={kto_mean_reward:.2f}")

    # ── KTODiffusionController test ──
    print(f"\nRunning KTODiffusionController with NoiseModel + margin={GUIDANCE_MARGIN}")
    print(f"(zero thrust after KTO plan exhausted)\n")

    results = []

    # Save frames for a few seeds
    FRAME_SEEDS = seeds[:3]
    frame_env = gym.make("LL-test-noise", render_mode="rgb_array", continuous=True)
    for fs in FRAME_SEEDS:
        reward, landed, rms = run_episode(
            frame_env, fs, save_frames=True, frame_dir=f"./frames/seed_{fs}",
        )
        results.append((reward, landed, rms))
        status = "LANDED" if landed else "FAILED"
        print(f"  Seed {fs}: {status}  reward={reward:.2f}  rms={rms:.3f}m")
    frame_env.close()

    # Remaining episodes headless
    env = gym.make("LL-test-noise", render_mode=None, continuous=True)
    t0 = time.time()
    for i in range(len(FRAME_SEEDS), N_EPISODES):
        seed = seeds[i]
        reward, landed, rms = run_episode(env, seed)
        results.append((reward, landed, rms))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - len(FRAME_SEEDS)) / elapsed
            print(f"  [{i + 1}/{N_EPISODES}]  {rate:.1f} ep/s  "
                  f"land_rate={np.mean([r[1] for r in results]):.0%}")
    env.close()

    # Summary
    rewards = [r[0] for r in results]
    landings = [r[1] for r in results]
    rms_errors = [r[2] for r in results]
    land_rate = np.mean(landings)
    mean_reward = np.mean(rewards)
    mean_rms = np.mean(rms_errors)
    delta = land_rate - kto_land_rate

    print(f"\n{'='*60}")
    print(f"Results ({N_EPISODES} episodes)")
    print(f"{'='*60}")
    print(f"KTO baseline:         land_rate={kto_land_rate:.0%}  reward={kto_mean_reward:.2f}")
    print(f"KTODiffusionController: land_rate={land_rate:.0%}  reward={mean_reward:.2f}")
    print(f"Delta:                {delta:+.0%}")
    print(f"Mean tracking RMS:    {mean_rms:.3f}m  (target: < 0.5m)")
    print(f"{'='*60}")

    passed = True
    if land_rate < kto_land_rate - 0.05:
        print(f"FAIL: Landing rate {land_rate:.0%} is >5% below KTO baseline {kto_land_rate:.0%}")
        passed = False
    if mean_rms > 0.5:
        print(f"FAIL: Mean tracking RMS {mean_rms:.3f}m > 0.5m")
        passed = False
    if passed:
        print("PASS: Pipeline within 5% of KTO baseline, tracking RMS < 0.5m")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
