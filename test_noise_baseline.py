"""Step 0 smoke test: NoiseModel + guidance_margin=0.001 reproduces KTO baseline.

Validates DiffusionAction PD tracking + guidance margin pipeline.
With margin≈0, the model output is ignored and DiffusionAction tracks the
KTO controller's reference position exactly via PD control.

Pass criteria:
  - Landing rate within 5% of KTO baseline (run KTO first to measure)
  - Mean position tracking RMS < 0.5m
"""

import os
import sys
import time

import gymnasium as gym
import numpy as np
from PIL import Image

from lunar_lander import LunarLander, KTOController, DT, TIMEOUT, heuristic
from diffusion_controller import (
    DiffusionController, NoiseModel,
    LanderState, ObstacleRelative, WaypointTarget, GuidanceAction, Outcome,
)
import solver


GUIDANCE_MARGIN = 0.001
N_EPISODES = 100
SEED_OFFSET = 2000
SAVE_FRAMES_SEED = 2000  # save frames for this one seed


def get_world_state(env):
    """Extract (x, y, theta) from Box2D body."""
    uw = env.unwrapped
    L = uw.lander
    return (L.position.x, L.position.y, L.angle)


def get_world_velocity(env):
    """Extract (vx, vy, omega) from Box2D body."""
    uw = env.unwrapped
    L = uw.lander
    return (L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity)


def run_episode(env, seed, noise_model, save_frames=False, frame_dir=None):
    """Run one episode: KTO expert provides guidance, DiffusionController acts."""
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    uw.lander.linearVelocity = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0

    # Create KTO expert
    kto = KTOController(env, time_budget=5.0)

    q_prev = get_world_state(env)
    total_reward = 0.0
    done = False
    step_idx = 0
    tracking_errors = []
    frames = []

    while not done:
        q_now = get_world_state(env)
        vel = get_world_velocity(env)
        t_sim = uw.elapsed_s

        # Check if KTO still has plan steps BEFORE calling step
        in_plan = kto.idx < kto.n_steps

        # KTO expert action (advances kto.idx)
        kto_action = kto.step(env)

        if in_plan:
            # Use DiffusionController with KTO plan as guidance
            # ref_idx is the step we just consumed (kto.idx was incremented)
            ref_idx = min(kto.idx - 1, kto.n_steps - 1)
            p = kto.plan
            guidance_q = (
                float(p["x"][ref_idx]),
                float(p["y"][ref_idx]),
                float(p["theta"][ref_idx]),
            )
            guidance_qp = (
                float(p["vx"][ref_idx]),
                float(p["vy"][ref_idx]),
                float(p["omega"][ref_idx]),
            )
            guidance_qpp = (
                float(p["ax"][ref_idx]),
                float(p["ay"][ref_idx]),
                float(p["alpha"][ref_idx]),
            )

            # Build waypoint: pad-relative target
            pad_x, pad_y = solver.PAD_X, solver.PAD_Y
            dq = (pad_x - q_now[0], pad_y - q_now[1], 0.0 - q_now[2])
            dq_prime = (0.0 - vel[0], -0.5 - vel[1], 0.0 - vel[2])

            action_obj = DiffusionController(
                timeout=TIMEOUT,
                t_obs_cmd_latency=DT,
                lander_state=LanderState(t_sim=t_sim, q_now=q_now, q_prev=q_prev),
                obstacle=ObstacleRelative(dx=0.0, dy=0.0, r=0.0),
                waypoint=WaypointTarget(dq=dq, dq_prime=dq_prime),
                guidance=GuidanceAction(q=guidance_q, q_prime=guidance_qp, q_double_prime=guidance_qpp),
                guidance_margin=GUIDANCE_MARGIN,
                outcome=Outcome.SUCCESS,
                action_horizon=1.5,
                model=noise_model,
            )

            tv, th = action_obj.step(t=DT, q_now=q_now, v_now=vel)
            action_out = np.array([tv, th], dtype=np.float32)

            # Track position error vs KTO reference
            pos_err = np.sqrt(
                (q_now[0] - guidance_q[0])**2 + (q_now[1] - guidance_q[1])**2
            )
            tracking_errors.append(pos_err)
        else:
            # KTO exhausted plan — use heuristic directly (same as KTOController)
            action_out = kto_action

        # Save frame if requested
        if save_frames and hasattr(env, 'render'):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        obs, reward, term, trunc, _ = env.step(action_out)
        total_reward += reward
        q_prev = q_now
        done = term or trunc
        step_idx += 1

    landed = (
        not uw.game_over
        and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
    )

    # Save frames to disk
    if save_frames and frames and frame_dir:
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(os.path.join(frame_dir, f"frame_{i:04d}.png"))
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
    # Register env
    gym.register(
        id="LL-test-noise",
        entry_point="lunar_lander:LunarLander",
        max_episode_steps=1000,
    )

    seeds = [SEED_OFFSET + i for i in range(N_EPISODES)]
    noise_model = NoiseModel()

    # ── KTO baseline ──
    print(f"Running KTO baseline ({N_EPISODES} episodes)...")
    env_kto = gym.make("LL-test-noise", render_mode=None, continuous=True)
    kto_land_rate, kto_mean_reward = run_kto_baseline(env_kto, seeds)
    env_kto.close()
    print(f"  KTO baseline: land_rate={kto_land_rate:.0%}  reward={kto_mean_reward:.2f}")

    # ── Diffusion pipeline test ──
    print(f"\nRunning DiffusionController with NoiseModel + guidance_margin={GUIDANCE_MARGIN}")
    print(f"Saving frames for seed={SAVE_FRAMES_SEED}\n")

    results = []

    # First run: save frames
    frame_env = gym.make("LL-test-noise", render_mode="rgb_array", continuous=True)
    reward, landed, rms = run_episode(
        frame_env, SAVE_FRAMES_SEED, noise_model,
        save_frames=True, frame_dir="./frames",
    )
    frame_env.close()
    results.append((reward, landed, rms))
    status = "LANDED" if landed else "FAILED"
    print(f"  Seed {SAVE_FRAMES_SEED}: {status}  reward={reward:.2f}  rms={rms:.3f}m")

    # Remaining episodes headless
    env = gym.make("LL-test-noise", render_mode=None, continuous=True)
    t0 = time.time()
    for i in range(1, N_EPISODES):
        seed = seeds[i]
        reward, landed, rms = run_episode(env, seed, noise_model)
        results.append((reward, landed, rms))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
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
    print(f"KTO baseline:       land_rate={kto_land_rate:.0%}  reward={kto_mean_reward:.2f}")
    print(f"Diffusion pipeline: land_rate={land_rate:.0%}  reward={mean_reward:.2f}")
    print(f"Delta:              {delta:+.0%}")
    print(f"Mean tracking RMS:  {mean_rms:.3f}m  (target: < 0.5m)")
    print(f"{'='*60}")

    # Pass/fail: within 5% of KTO baseline + tracking RMS < 0.5m
    passed = True
    if land_rate < kto_land_rate - 0.05:
        print(f"FAIL: Landing rate {land_rate:.0%} is >{5}% below KTO baseline {kto_land_rate:.0%}")
        passed = False
    if mean_rms > 0.5:
        print(f"FAIL: Mean tracking RMS {mean_rms:.3f}m > 0.5m")
        passed = False
    if passed:
        print("PASS: Pipeline within 5% of KTO baseline, tracking RMS < 0.5m")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
