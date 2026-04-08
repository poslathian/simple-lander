"""Fast parallel rollout collection on Modal with KTO plan caching.

The KTO trajectory solve is the bottleneck (~2-5s per episode vs ~0.03s
for the actual rollout). This module:
  1. Pre-computes KTO plans for all seeds in one batch (cacheable)
  2. Runs rollouts in parallel across Modal containers using cached plans
  3. Returns serialized episode + frame data for local DB ingestion

Usage (local):
    from modal_rollout import collect_parallel_modal
    results = collect_parallel_modal(checkpoint_bytes, seeds, margin, ...)

Usage (standalone test):
    .venv/bin/python modal_rollout.py --test
"""

import os
import pickle
import sys
import time

import modal
import numpy as np

app = modal.App("position-dagger-rollout")

# ── Modal image ──────────────────────────────────────────────────────────

rollout_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("swig", "build-essential")
    .pip_install(
        "torch", "numpy", "scipy", "gymnasium[box2d]", "swig",
        "drake>=1.51",
    )
)


# ── Source file helpers ──────────────────────────────────────────────────

def read_source_files():
    """Read all source files needed on Modal containers."""
    files = {}
    for name in [
        "model.py", "diffusion_controller.py", "lunar_lander.py",
        "solver.py", "thrust_spline.py", "eval.py", "collect.py",
        "guidance_controller.py",
    ]:
        if os.path.exists(name):
            with open(name, "rb") as f:
                files[name] = f.read()
    return files


def _write_sources(files, workdir="/root/work"):
    """Write source files into container workdir."""
    import os as _os
    _os.makedirs(workdir, exist_ok=True)
    for name, content in files.items():
        with open(f"{workdir}/{name}", "wb") as f:
            f.write(content)
    return workdir


# ── Remote: batch KTO solve ─────────────────────────────────────────────

@app.function(
    image=rollout_image,
    timeout=600,
    cpu=2,
    memory=4096,
)
def solve_kto_batch(
    source_files: dict,
    seeds: list[int],
) -> dict:
    """Solve KTO plans for a batch of seeds. Returns {seed: plan_data}.

    Each plan_data is a pickle-safe dict with plan arrays and metadata.
    This is the expensive step (~2-5s per seed) that we want to cache.
    """
    import sys
    workdir = _write_sources(source_files)
    sys.path.insert(0, workdir)
    os.chdir(workdir)

    import gymnasium as gym
    import numpy as np
    from lunar_lander import LunarLander, KTOController

    try:
        gym.register(id="LL-kto", entry_point="lunar_lander:LunarLander",
                     max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-kto", render_mode=None, continuous=True)

    results = {}
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        t0 = time.time()
        kto = KTOController(env, time_budget=5.0)
        solve_time = time.time() - t0

        # Serialize the plan — convert all arrays to numpy for pickling
        plan_data = {
            "plan": {k: np.array(v, dtype=np.float64) for k, v in kto.plan.items()},
            "plan_times": np.array(kto.plan_times, dtype=np.float64),
            "n_steps": kto.n_steps,
            "solve_time": solve_time,
        }
        results[seed] = plan_data

    env.close()
    return results


# ── Remote: rollout with cached KTO plan ─────────────────────────────────

@app.function(
    image=rollout_image,
    timeout=600,
    cpu=2,
    memory=4096,
)
def rollout_with_cache(
    source_files: dict,
    checkpoint_bytes: bytes,
    seeds_and_plans: bytes,  # pickled list of (seed, plan_data)
    margin_mean: float,
    outcome_cond: int = 1,
    hidden: int = 256,
    n_blocks: int = 6,
) -> list[dict]:
    """Run rollouts using pre-cached KTO plans. Much faster than solving per-episode.

    Returns list of episode dicts with embedded frame data (bytes for transport).
    """
    import sys
    workdir = _write_sources(source_files)
    sys.path.insert(0, workdir)
    os.chdir(workdir)

    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.interpolate import make_lsq_spline

    from model import DiffusionMLP, CosineSchedule, DDIMSampler
    from model import X_DIM, COND_DIM, STATE_DIM, CFG_START, CFG_END, N_CPS, N_CHANNELS
    from diffusion_controller import (
        KTODiffusionController, Outcome,
        _build_cond, ObstacleRelative, WaypointTarget,
        DT, DEGREE, _make_position_spline,
    )
    import solver

    # Deserialize inputs
    seed_plan_list = pickle.loads(seeds_and_plans)

    # Load model
    with open("/tmp/model.pt", "wb") as f:
        f.write(checkpoint_bytes)
    ckpt = torch.load("/tmp/model.pt", weights_only=False)
    model = DiffusionMLP(hidden=hidden, n_blocks=n_blocks)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        pass  # fresh model for size mismatch
    model.eval()
    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32)
    schedule = CosineSchedule(T=ckpt.get("T", 100))
    sampler = DDIMSampler(model, schedule, n_steps=10)

    class _Model:
        def predict(self, cond, outcome, guidance_scale=2.0):
            full = np.zeros(COND_DIM, dtype=np.float32)
            full[:STATE_DIM] = cond[:STATE_DIM]
            full[CFG_START] = float(outcome)
            ct = torch.tensor(full, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                x_norm = sampler.sample_cfg(ct, guidance_scale=guidance_scale)
            x_raw = (x_norm * x_std + x_mean).squeeze(0).numpy()
            cps = x_raw.reshape(N_CPS, N_CHANNELS)
            cps[0] = [0, 0, 0]
            return cps

    live_model = _Model()
    outcome_enum = Outcome.SUCCESS if outcome_cond == 1 else Outcome.FAIL

    # Fit KTO window helper (inline to avoid import issues)
    def _fit_kto_window(plan, idx, action_horizon, dt=DT):
        n_steps = int(round(action_horizon / dt))
        end_idx = min(idx + n_steps, len(plan["x"]) - 1)
        if end_idx <= idx:
            return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        n = end_idx - idx
        t = np.linspace(0, (n - 1) * dt, n)
        x0 = float(plan["x"][idx])
        y0 = float(plan["y"][idx])
        th0 = float(plan["theta"][idx])
        x_rel = np.array(plan["x"][idx:end_idx], dtype=np.float64) - x0
        y_rel = np.array(plan["y"][idx:end_idx], dtype=np.float64) - y0
        th_rel = np.array(plan["theta"][idx:end_idx], dtype=np.float64) - th0
        if len(t) < N_CPS:
            return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        duration = t[-1]
        if duration < 1e-6:
            return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        n_internal = N_CPS - DEGREE + 1
        internal = np.linspace(0, duration, n_internal)
        knots = np.concatenate([
            np.full(DEGREE + 1, 0.0), internal[1:-1], np.full(DEGREE + 1, duration),
        ])
        cps = np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        for ch, vals in enumerate([x_rel, y_rel, th_rel]):
            try:
                spline = make_lsq_spline(t, vals, knots, k=DEGREE)
                cps[:, ch] = spline.c[:N_CPS]
            except Exception:
                pass
        return cps

    try:
        gym.register(id="LL-cached", entry_point="lunar_lander:LunarLander",
                     max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-cached", render_mode=None, continuous=True)

    results = []
    for seed, plan_data in seed_plan_list:
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        # Build controller with cached KTO plan (skip the expensive solve)
        ctrl = KTODiffusionController(
            env, model=live_model, target_frequency=3.0,
            action_horizon=1.5, outcome=outcome_enum,
        )
        # Inject cached plan instead of calling warm_start()
        ctrl._kto = type("CachedKTO", (), {
            "plan": plan_data["plan"],
            "plan_times": plan_data["plan_times"],
            "n_steps": plan_data["n_steps"],
        })()
        ctrl._kto_t0 = uw.elapsed_s
        ctrl._kto_duration = plan_data["n_steps"] * DT

        frames = []
        total_reward = 0.0
        done = False
        step_idx = 0
        last_inf = -999
        spi = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))

        while not done:
            t_sim = uw.elapsed_s
            L = uw.lander

            if step_idx - last_inf >= spi:
                q_now = (L.position.x, L.position.y, L.angle)
                vel = (L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity)
                q_prev = ctrl._last_inference_q if ctrl._last_inference_q else q_now
                dq = (solver.PAD_X - q_now[0], solver.PAD_Y - q_now[1], 0.0 - q_now[2])
                dq_prime = (0.0 - vel[0], -0.5 - vel[1], 0.0 - vel[2])
                kto_ref = ctrl._get_kto_ref(t_sim)
                guidance_q = kto_ref["q"] if kto_ref else q_now

                model_input = _build_cond(
                    t_obs_cmd_latency=DT, q_now=q_now, q_prev=q_prev,
                    obstacle=ObstacleRelative(0.0, 0.0, 0.0),
                    waypoint=WaypointTarget(dq=dq, dq_prime=dq_prime),
                    guidance_q=guidance_q, action_horizon=ctrl.action_horizon,
                )

                ctrl.inference()
                last_inf = step_idx

                model_out = np.zeros((N_CPS, N_CHANNELS), dtype=np.float32)
                if ctrl._diff_splines is not None:
                    for ch in range(N_CHANNELS):
                        model_out[:, ch] = ctrl._diff_splines[ch].c[:N_CPS]

                kto_idx = int(round((t_sim - ctrl._kto_t0) / DT))
                actual_cps = _fit_kto_window(ctrl._kto.plan, kto_idx, ctrl.action_horizon)

                frames.append({
                    "t_sim": t_sim,
                    "model_input": model_input.tobytes(),
                    "model_output_cps": model_out.flatten().tobytes(),
                    "actual_tracked_cps": actual_cps.astype(np.float32).flatten().tobytes(),
                })

            actual_margin = float(np.clip(abs(np.random.normal(margin_mean, 0.05)), 0.001, 1.0))
            tv, th = ctrl.get_action(guidance_margin=actual_margin)
            obs, reward, term, trunc, _ = env.step(np.array([tv, th], dtype=np.float32))
            total_reward += reward
            done = term or trunc
            step_idx += 1

        landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)

        results.append({
            "seed": seed,
            "outcome": 1 if landed else -1,
            "sim_duration": uw.elapsed_s,
            "reward": total_reward,
            "landed": landed,
            "frames": frames,
            "margin": margin_mean,
        })

    env.close()
    return results


# ── Local orchestration ──────────────────────────────────────────────────

def collect_parallel_modal(
    checkpoint_path: str,
    seeds: list[int],
    margin_mean: float,
    hidden: int = 256,
    n_blocks: int = 6,
    kto_batch_size: int = 20,
    rollout_batch_size: int = 10,
) -> list[dict]:
    """Collect episodes via Modal with KTO caching.

    1. Batch-solve KTO plans across containers (the expensive part)
    2. Distribute cached plans + rollouts across containers (fast part)

    Returns list of episode result dicts.
    """
    source_files = read_source_files()
    with open(checkpoint_path, "rb") as f:
        ckpt_bytes = f.read()

    # Phase 1: Parallel KTO solve
    print(f"  Phase 1: Solving KTO plans for {len(seeds)} seeds...")
    t0 = time.time()

    kto_batches = [seeds[i:i+kto_batch_size]
                   for i in range(0, len(seeds), kto_batch_size)]
    kto_args = [(source_files, batch) for batch in kto_batches]

    all_plans = {}  # seed -> plan_data
    for batch_result in solve_kto_batch.starmap(kto_args):
        all_plans.update(batch_result)

    t_kto = time.time() - t0
    print(f"    {len(all_plans)} plans solved in {t_kto:.1f}s "
          f"({t_kto/max(len(all_plans),1):.2f}s/plan effective)")

    # Phase 2: Parallel rollouts with cached plans
    print(f"  Phase 2: Running {len(seeds)} rollouts with cached plans...")
    t1 = time.time()

    seed_plan_pairs = [(s, all_plans[s]) for s in seeds if s in all_plans]
    rollout_batches = [seed_plan_pairs[i:i+rollout_batch_size]
                       for i in range(0, len(seed_plan_pairs), rollout_batch_size)]

    rollout_args = [
        (source_files, ckpt_bytes, pickle.dumps(batch), margin_mean, 1, hidden, n_blocks)
        for batch in rollout_batches
    ]

    all_results = []
    for batch_result in rollout_with_cache.starmap(rollout_args):
        all_results.extend(batch_result)

    t_rollout = time.time() - t1
    n_landed = sum(1 for r in all_results if r["landed"])
    print(f"    {len(all_results)} episodes in {t_rollout:.1f}s "
          f"({n_landed} landed, {len(all_results)-n_landed} failed)")

    return all_results


# ── Local-only functions for testing without Modal ───────────────────────

def solve_kto_local(seeds):
    """Solve KTO plans locally. Same output format as solve_kto_batch."""
    import gymnasium as gym
    from lunar_lander import LunarLander, KTOController

    try:
        gym.register(id="LL-kto-local", entry_point="lunar_lander:LunarLander",
                     max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-kto-local", render_mode=None, continuous=True)

    results = {}
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        t0 = time.time()
        kto = KTOController(env, time_budget=5.0)
        solve_time = time.time() - t0

        plan_data = {
            "plan": {k: np.array(v, dtype=np.float64) for k, v in kto.plan.items()},
            "plan_times": np.array(kto.plan_times, dtype=np.float64),
            "n_steps": kto.n_steps,
            "solve_time": solve_time,
        }
        results[seed] = plan_data

    env.close()
    return results


def rollout_with_cache_local(
    checkpoint_path, seed_plan_list, margin_mean,
    hidden=256, n_blocks=6, outcome_cond=1,
):
    """Run rollouts locally using cached KTO plans. Same output as remote version."""
    import gymnasium as gym
    import torch
    from scipy.interpolate import make_lsq_spline

    from model import DiffusionMLP, CosineSchedule, DDIMSampler
    from model import COND_DIM, STATE_DIM, CFG_START, N_CPS, N_CHANNELS
    from diffusion_controller import (
        KTODiffusionController, Outcome,
        _build_cond, ObstacleRelative, WaypointTarget,
        DT, DEGREE,
    )
    import solver

    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = DiffusionMLP(hidden=hidden, n_blocks=n_blocks)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        pass
    model.eval()
    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32)
    schedule = CosineSchedule(T=ckpt.get("T", 100))
    sampler = DDIMSampler(model, schedule, n_steps=10)

    class _Model:
        def predict(self, cond, outcome, guidance_scale=2.0):
            full = np.zeros(COND_DIM, dtype=np.float32)
            full[:STATE_DIM] = cond[:STATE_DIM]
            full[CFG_START] = float(outcome)
            ct = torch.tensor(full, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                x_norm = sampler.sample_cfg(ct, guidance_scale=guidance_scale)
            x_raw = (x_norm * x_std + x_mean).squeeze(0).numpy()
            cps = x_raw.reshape(N_CPS, N_CHANNELS)
            cps[0] = [0, 0, 0]
            return cps

    live_model = _Model()
    outcome_enum = Outcome.SUCCESS if outcome_cond == 1 else Outcome.FAIL

    def _fit_kto_window(plan, idx, action_horizon, dt=DT):
        n_steps = int(round(action_horizon / dt))
        end_idx = min(idx + n_steps, len(plan["x"]) - 1)
        if end_idx <= idx:
            return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        n = end_idx - idx
        t = np.linspace(0, (n - 1) * dt, n)
        x0, y0, th0 = float(plan["x"][idx]), float(plan["y"][idx]), float(plan["theta"][idx])
        x_rel = np.array(plan["x"][idx:end_idx], dtype=np.float64) - x0
        y_rel = np.array(plan["y"][idx:end_idx], dtype=np.float64) - y0
        th_rel = np.array(plan["theta"][idx:end_idx], dtype=np.float64) - th0
        if len(t) < N_CPS:
            return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        duration = t[-1]
        if duration < 1e-6:
            return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        n_internal = N_CPS - DEGREE + 1
        internal = np.linspace(0, duration, n_internal)
        knots = np.concatenate([
            np.full(DEGREE + 1, 0.0), internal[1:-1], np.full(DEGREE + 1, duration),
        ])
        cps = np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        for ch, vals in enumerate([x_rel, y_rel, th_rel]):
            try:
                spline = make_lsq_spline(t, vals, knots, k=DEGREE)
                cps[:, ch] = spline.c[:N_CPS]
            except Exception:
                pass
        return cps

    try:
        gym.register(id="LL-cache-local", entry_point="lunar_lander:LunarLander",
                     max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-cache-local", render_mode=None, continuous=True)

    results = []
    for seed, plan_data in seed_plan_list:
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        ctrl = KTODiffusionController(
            env, model=live_model, target_frequency=3.0,
            action_horizon=1.5, outcome=outcome_enum,
        )
        # Inject cached plan
        ctrl._kto = type("CachedKTO", (), {
            "plan": plan_data["plan"],
            "plan_times": plan_data["plan_times"],
            "n_steps": plan_data["n_steps"],
        })()
        ctrl._kto_t0 = uw.elapsed_s
        ctrl._kto_duration = plan_data["n_steps"] * DT

        frames = []
        total_reward = 0.0
        done = False
        step_idx = 0
        last_inf = -999
        spi = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))

        while not done:
            t_sim = uw.elapsed_s
            L = uw.lander

            if step_idx - last_inf >= spi:
                q_now = (L.position.x, L.position.y, L.angle)
                vel = (L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity)
                q_prev = ctrl._last_inference_q if ctrl._last_inference_q else q_now
                dq = (solver.PAD_X - q_now[0], solver.PAD_Y - q_now[1], 0.0 - q_now[2])
                dq_prime = (0.0 - vel[0], -0.5 - vel[1], 0.0 - vel[2])
                kto_ref = ctrl._get_kto_ref(t_sim)
                guidance_q = kto_ref["q"] if kto_ref else q_now

                model_input = _build_cond(
                    t_obs_cmd_latency=DT, q_now=q_now, q_prev=q_prev,
                    obstacle=ObstacleRelative(0.0, 0.0, 0.0),
                    waypoint=WaypointTarget(dq=dq, dq_prime=dq_prime),
                    guidance_q=guidance_q, action_horizon=ctrl.action_horizon,
                )

                ctrl.inference()
                last_inf = step_idx

                model_out = np.zeros((N_CPS, N_CHANNELS), dtype=np.float32)
                if ctrl._diff_splines is not None:
                    for ch in range(N_CHANNELS):
                        model_out[:, ch] = ctrl._diff_splines[ch].c[:N_CPS]

                kto_idx = int(round((t_sim - ctrl._kto_t0) / DT))
                actual_cps = _fit_kto_window(ctrl._kto.plan, kto_idx, ctrl.action_horizon)

                frames.append({
                    "t_sim": t_sim,
                    "model_input": model_input.tobytes(),
                    "model_output_cps": model_out.flatten().tobytes(),
                    "actual_tracked_cps": actual_cps.astype(np.float32).flatten().tobytes(),
                })

            actual_margin = float(np.clip(abs(np.random.normal(margin_mean, 0.05)), 0.001, 1.0))
            tv, th = ctrl.get_action(guidance_margin=actual_margin)
            obs, reward, term, trunc, _ = env.step(np.array([tv, th], dtype=np.float32))
            total_reward += reward
            done = term or trunc
            step_idx += 1

        landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)

        results.append({
            "seed": seed,
            "outcome": 1 if landed else -1,
            "sim_duration": uw.elapsed_s,
            "reward": total_reward,
            "landed": landed,
            "frames": frames,
            "margin": margin_mean,
        })

    env.close()
    return results


# ── CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run local tests")
    parser.add_argument("--test-modal", action="store_true", help="Run Modal integration test")
    parser.add_argument("--checkpoint", default="position_model.pt")
    args = parser.parse_args()

    if args.test:
        print("Running local tests...")
        # Imported in test_modal_rollout.py, but can also run inline
        from test_modal_rollout import run_all_tests
        run_all_tests(args.checkpoint)
    elif args.test_modal:
        print("Running Modal integration test...")
        seeds = list(range(7000, 7010))
        with app.run():
            results = collect_parallel_modal(
                args.checkpoint, seeds, margin_mean=0.2,
            )
        n_landed = sum(1 for r in results if r["landed"])
        print(f"\nModal test: {n_landed}/{len(results)} landed")
        for r in results:
            status = "LANDED" if r["landed"] else "FAILED"
            print(f"  seed={r['seed']} {status} frames={len(r['frames'])} "
                  f"sim_t={r['sim_duration']:.1f}s")
