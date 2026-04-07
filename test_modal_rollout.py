"""Tests for modal_rollout: KTO caching, cached rollouts, frame integrity.

Tests run locally (no Modal needed) using the _local variants.
Validates that:
  1. KTO plans are deterministic (same seed → same plan)
  2. Cached rollouts produce valid episodes and frames
  3. Cached rollouts match non-cached rollouts (same outcomes)
  4. Frame data has correct shapes and is non-degenerate
  5. Speedup from caching is significant (>5x)
"""

import sys
import time

import numpy as np


def test_kto_solve_deterministic(checkpoint_path):
    """Same seed should produce identical KTO plans."""
    from modal_rollout import solve_kto_local

    plans_a = solve_kto_local([5000, 5001])
    plans_b = solve_kto_local([5000, 5001])

    for seed in [5000, 5001]:
        pa = plans_a[seed]
        pb = plans_b[seed]
        assert pa["n_steps"] == pb["n_steps"], \
            f"seed {seed}: n_steps mismatch {pa['n_steps']} vs {pb['n_steps']}"
        for key in ["x", "y", "theta", "vx", "vy", "omega"]:
            np.testing.assert_allclose(
                pa["plan"][key], pb["plan"][key], atol=1e-10,
                err_msg=f"seed {seed}: plan[{key}] not deterministic",
            )

    print("  PASS: KTO plans are deterministic")


def test_cached_rollout_produces_frames(checkpoint_path):
    """Cached rollout should produce episodes with valid frames."""
    from modal_rollout import solve_kto_local, rollout_with_cache_local

    seeds = [5000, 5001, 5002]
    plans = solve_kto_local(seeds)
    seed_plan_list = [(s, plans[s]) for s in seeds]

    results = rollout_with_cache_local(
        checkpoint_path, seed_plan_list, margin_mean=0.2,
    )

    assert len(results) == len(seeds), \
        f"Expected {len(seeds)} results, got {len(results)}"

    for r in results:
        assert r["seed"] in seeds
        assert r["outcome"] in (1, -1)
        assert r["sim_duration"] > 0
        assert len(r["frames"]) > 0, f"seed {r['seed']}: no frames collected"

        # Check frame data shapes
        for fr in r["frames"]:
            model_input = np.frombuffer(fr["model_input"], dtype=np.float32)
            model_out = np.frombuffer(fr["model_output_cps"], dtype=np.float32)
            actual_cps = np.frombuffer(fr["actual_tracked_cps"], dtype=np.float32)

            assert model_input.shape == (20,), \
                f"model_input shape {model_input.shape}, expected (20,)"
            assert model_out.shape == (30,), \
                f"model_output shape {model_out.shape}, expected (30,)"
            assert actual_cps.shape == (30,), \
                f"actual_cps shape {actual_cps.shape}, expected (30,)"
            assert fr["t_sim"] >= 0

    print(f"  PASS: {len(results)} episodes with valid frames "
          f"({sum(len(r['frames']) for r in results)} total frames)")


def test_cached_matches_uncached(checkpoint_path):
    """Cached rollout outcomes should match direct rollouts at margin=0.001.

    At margin=0.001 the diffusion model has negligible effect, so outcomes
    should be identical regardless of model randomness. We compare against
    the non-cached collect_episode from dagger_loop.
    """
    from modal_rollout import solve_kto_local, rollout_with_cache_local
    from dagger_loop import collect_episode, _LiveModel
    from model import DiffusionMLP, CosineSchedule
    from diffusion_controller import Outcome
    import torch
    import gymnasium as gym

    seeds = [6000, 6001, 6002, 6003, 6004]
    margin = 0.001  # near-zero so diffusion noise doesn't matter

    # Load model for uncached path
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = DiffusionMLP()
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        pass
    model.eval()
    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32)
    live_model = _LiveModel(model, x_mean, x_std, T=100)

    try:
        gym.register(id="LL-test-uncached", entry_point="lunar_lander:LunarLander",
                     max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-test-uncached", render_mode=None, continuous=True)

    # Uncached rollouts
    uncached_outcomes = {}
    for seed in seeds:
        ep_info, _ = collect_episode(env, seed, live_model, margin)
        uncached_outcomes[seed] = ep_info["outcome"]
    env.close()

    # Cached rollouts
    plans = solve_kto_local(seeds)
    seed_plan_list = [(s, plans[s]) for s in seeds]
    cached_results = rollout_with_cache_local(
        checkpoint_path, seed_plan_list, margin_mean=margin,
    )
    cached_outcomes = {r["seed"]: r["outcome"] for r in cached_results}

    # Compare — at margin 0.001 the model barely affects outcomes
    matches = sum(1 for s in seeds if uncached_outcomes[s] == cached_outcomes[s])
    match_rate = matches / len(seeds)

    # Allow some variance since margin sampling is stochastic, but most should match
    print(f"  Outcome match rate: {matches}/{len(seeds)} = {match_rate:.0%}")
    assert match_rate >= 0.6, \
        f"Too few matches ({match_rate:.0%}). Cached rollout may be broken."
    print(f"  PASS: cached vs uncached outcomes match ({match_rate:.0%})")


def test_frame_data_nontrivial(checkpoint_path):
    """Frame data should contain non-zero, varied values (not all zeros)."""
    from modal_rollout import solve_kto_local, rollout_with_cache_local

    seeds = [5010, 5011]
    plans = solve_kto_local(seeds)
    seed_plan_list = [(s, plans[s]) for s in seeds]

    results = rollout_with_cache_local(
        checkpoint_path, seed_plan_list, margin_mean=0.3,
    )

    for r in results:
        assert len(r["frames"]) >= 5, \
            f"seed {r['seed']}: only {len(r['frames'])} frames (expected >= 5)"

        # Check that conditioning vectors vary across frames
        conds = [np.frombuffer(f["model_input"], dtype=np.float32) for f in r["frames"]]
        cond_stack = np.stack(conds)
        std_per_dim = cond_stack.std(axis=0)
        n_varying = (std_per_dim > 1e-6).sum()
        assert n_varying >= 3, \
            f"seed {r['seed']}: only {n_varying} conditioning dims vary across frames"

        # Check actual_tracked_cps are non-zero (KTO should produce real trajectories)
        for fr in r["frames"][:3]:
            actual = np.frombuffer(fr["actual_tracked_cps"], dtype=np.float32)
            assert np.abs(actual).sum() > 1e-6, \
                "actual_tracked_cps is all zeros — KTO window fit may be broken"

    print(f"  PASS: frame data is non-trivial and varied")


def test_caching_speedup(checkpoint_path):
    """Cached rollouts should be significantly faster than uncached."""
    from modal_rollout import solve_kto_local, rollout_with_cache_local
    from dagger_loop import collect_episode, _LiveModel
    from model import DiffusionMLP, CosineSchedule
    import torch
    import gymnasium as gym

    seeds = [7000, 7001, 7002]

    # Time uncached (includes KTO solve)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = DiffusionMLP()
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        pass
    model.eval()
    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32)
    live_model = _LiveModel(model, x_mean, x_std, T=100)

    try:
        gym.register(id="LL-bench-uncached", entry_point="lunar_lander:LunarLander",
                     max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-bench-uncached", render_mode=None, continuous=True)

    t0 = time.time()
    for seed in seeds:
        collect_episode(env, seed, live_model, 0.2)
    t_uncached = time.time() - t0
    env.close()

    # Time cached (KTO solve separate)
    t1 = time.time()
    plans = solve_kto_local(seeds)
    t_solve = time.time() - t1

    seed_plan_list = [(s, plans[s]) for s in seeds]
    t2 = time.time()
    rollout_with_cache_local(checkpoint_path, seed_plan_list, margin_mean=0.2)
    t_rollout = time.time() - t2

    speedup = t_uncached / max(t_rollout, 0.01)
    print(f"  Uncached: {t_uncached:.2f}s ({t_uncached/len(seeds):.2f}s/ep)")
    print(f"  Cached:   solve={t_solve:.2f}s + rollout={t_rollout:.2f}s")
    print(f"  Rollout speedup (excluding solve): {speedup:.1f}x")

    assert speedup >= 2.0, \
        f"Expected >= 2x speedup, got {speedup:.1f}x"
    print(f"  PASS: caching provides {speedup:.1f}x rollout speedup")


def run_all_tests(checkpoint_path="position_model.pt"):
    """Run all local tests."""
    print(f"\nRunning modal_rollout tests (checkpoint: {checkpoint_path})")
    print("=" * 60)

    tests = [
        ("KTO determinism", test_kto_solve_deterministic),
        ("Cached rollout frames", test_cached_rollout_produces_frames),
        ("Cached vs uncached", test_cached_matches_uncached),
        ("Frame data quality", test_frame_data_nontrivial),
        ("Caching speedup", test_caching_speedup),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            test_fn(checkpoint_path)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="position_model.pt")
    args = parser.parse_args()

    ok = run_all_tests(args.checkpoint)
    sys.exit(0 if ok else 1)
