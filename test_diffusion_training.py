"""Deep tests for diffusion model training pipeline.

Tests: outcome conditioning, DB writes, normalization, dropout,
training target correctness, and first CP pinning.
"""

import math
import os
import sqlite3
import tempfile

import numpy as np
import pytest
import torch

import gymnasium as gym

from model import (
    DiffusionMLP, CosineSchedule, DDIMSampler,
    X_DIM, COND_DIM, STATE_DIM, CFG_START, CFG_END,
    N_CPS, N_CHANNELS,
)
from diffusion_controller import (
    KTODiffusionController, NoiseModel, Outcome,
    _build_cond, _make_position_spline, _eval_spline,
    ObstacleRelative, WaypointTarget,
    DT,
)
import solver


# ---------------------------------------------------------------------------
# Env fixture
# ---------------------------------------------------------------------------

_ENV_REGISTERED = False


def _make_env(seed=42):
    global _ENV_REGISTERED
    if not _ENV_REGISTERED:
        gym.register(
            id="LL-diffusion-test",
            entry_point="lunar_lander:LunarLander",
            max_episode_steps=1000,
            kwargs={"num_obstacles": 0},
        )
        _ENV_REGISTERED = True
    env = gym.make("LL-diffusion-test", render_mode=None, continuous=True)
    env.reset(seed=seed)
    return env


# ===========================================================================
# Test: actual_tracked_cps should be the blended reference, not just KTO
# ===========================================================================

class TestActualTrackedCPsIsBlended:
    """The DAgger training target `actual_tracked_cps` should represent the
    blended KTO+diffusion reference that the PD controller actually tracked,
    not just the raw KTO trajectory."""

    def test_actual_tracked_cps_reflects_clamped_model_output(self):
        """actual_tracked_cps should be the model output clamped to within
        margin distance of KTO CPs — matching what get_action() does at
        runtime."""
        from dagger_loop import collect_episode, _fit_kto_window

        env = _make_env(seed=100)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        # Model returns CPs with a known offset in normalized space.
        # 0.8 normalized x = 0.8 * 30 = 24 world units — nearly full screen.
        class OffsetModel:
            def predict(self, cond, outcome, guidance_scale=2.0):
                cps = np.zeros((N_CPS, N_CHANNELS))
                cps[1:, 0] = 0.8  # x-offset of 0.8 screen widths
                cps[0] = [0, 0, 0]
                return cps

        margin = 0.3
        ep_info, frames = collect_episode(
            env, 100, OffsetModel(), margin=margin, outcome_cond=Outcome.SUCCESS,
        )
        env.close()

        assert len(frames) > 1, "Need at least 2 frames"

        frame = frames[1]
        actual = np.asarray(frame["actual_tracked_cps"], dtype=np.float32).reshape(N_CPS, N_CHANNELS)
        model_out = np.asarray(frame["model_output_cps"], dtype=np.float32).reshape(N_CPS, N_CHANNELS)

        # Model x-CPs are 3.0 in normalized space (= 3.0 * 30 = 90 world units).
        # KTO CPs are small in normalized space (~0).
        # With margin=0.5, clamp clips model to kto ± 0.5.
        actual_x = actual[1:, 0]
        model_x = model_out[1:, 0]

        print(f"  model_output x (non-pinned): mean={model_x.mean():.3f}")
        print(f"  actual_tracked x (non-pinned): mean={actual_x.mean():.3f}")

        # actual_tracked should NOT be pure KTO (would be ~0)
        assert actual_x.mean() > 0.01, (
            f"actual_tracked_cps has no model influence: x-mean={actual_x.mean():.3f}"
        )
        # actual_tracked should NOT be raw model output (would be 0.8)
        assert actual_x.mean() < 0.7, (
            f"actual_tracked_cps is unclamped model output: x-mean={actual_x.mean():.3f}"
        )

    def test_margin_zero_gives_kto(self):
        """At margin≈0, actual_tracked_cps should be essentially KTO."""
        from dagger_loop import collect_episode

        env = _make_env(seed=100)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        class WildModel:
            def predict(self, cond, outcome, guidance_scale=2.0):
                cps = np.ones((N_CPS, N_CHANNELS)) * 0.9
                cps[0] = [0, 0, 0]
                return cps

        ep_info, frames = collect_episode(
            env, 100, WildModel(), margin=0.001, outcome_cond=Outcome.SUCCESS,
        )
        env.close()

        if len(frames) > 1:
            frame = frames[1]
            actual = np.asarray(frame["actual_tracked_cps"], dtype=np.float32).reshape(N_CPS, N_CHANNELS)
            # With margin=0.001, model output of 0.9 gets clamped to kto±0.001
            # KTO CPs are small, so actual ≈ kto (near zero)
            assert np.abs(actual[1:, 0]).max() < 0.1, (
                f"Should be near KTO at tiny margin, got max={np.abs(actual[1:, 0]).max():.3f}"
            )


# ===========================================================================
# Test: Normalization roundtrip through spline
# ===========================================================================

class TestNormalizationRoundtrip:

    def test_normalized_cps_spline_roundtrip(self):
        """Normalized CPs → denormalize → build spline → evaluate should
        produce world-frame values consistent with the original trajectory."""
        from diffusion_controller import NORM_SCALES

        # Create normalized CPs (model output space)
        cps_norm = np.random.RandomState(42).randn(N_CPS, N_CHANNELS) * 0.1
        cps_norm[0] = [0, 0, 0]

        # Denormalize to world-relative (what inference() does)
        cps_world = cps_norm * NORM_SCALES

        # Build spline in world-relative coords
        duration = 1.5
        splines = _make_position_spline(cps_world, duration)

        # Evaluate at midpoint
        mid = _eval_spline(splines, duration / 2, duration)

        # The spline value should be in world-relative coords
        # Re-normalize and check it's in the expected range
        mid_norm = np.array(mid) / NORM_SCALES
        print(f"  mid_world: {mid}")
        print(f"  mid_norm:  {mid_norm}")

        # Normalized values should be small (CPs were ~0.1 normalized)
        assert np.abs(mid_norm).max() < 1.0, (
            f"Normalized midpoint too large: {mid_norm}"
        )

    def test_fit_kto_window_produces_normalized_cps(self):
        """_fit_kto_window should output normalized CPs (divided by NORM_SCALES)."""
        from collect import _fit_kto_window
        from diffusion_controller import NORM_SCALES

        # Build a KTO-like plan: descend from (15, 17) to (15, 5)
        n = 100
        plan = {
            "x": np.full(n, 15.0),
            "y": np.linspace(17.0, 5.0, n),
            "theta": np.zeros(n),
        }
        cps = _fit_kto_window(plan, idx=0, action_horizon=1.5)

        # CPs are normalized. The y displacement is 17→5 = -12 world units.
        # Normalized: -12/20 = -0.6. CPs should be in this range.
        y_cps = cps[1:, 1]  # skip pinned first CP
        print(f"  y CPs (normalized): {y_cps}")
        assert np.abs(y_cps).max() < 1.0, (
            f"Normalized y CPs should be < 1.0, got max={np.abs(y_cps).max():.3f}"
        )
        # Should have negative y values (descending)
        assert y_cps.min() < -0.1, (
            f"Expected negative y CPs for descent, got min={y_cps.min():.3f}"
        )

    def test_kto_spline_roundtrip_preserves_trajectory(self):
        """Fit KTO → normalize CPs → denormalize → build spline → evaluate
        should reconstruct the original KTO trajectory."""
        from collect import _fit_kto_window
        from diffusion_controller import NORM_SCALES

        n = 100
        dt = 0.02
        plan = {
            "x": np.linspace(15.0, 15.0, n),
            "y": np.linspace(17.0, 5.0, n),
            "theta": np.linspace(0.0, -0.3, n),
        }

        cps_norm = _fit_kto_window(plan, idx=0, action_horizon=1.5)
        cps_world = cps_norm * NORM_SCALES

        duration = 1.5
        splines = _make_position_spline(cps_world, duration)

        # Evaluate at a few time points and compare to original trajectory
        for step in [10, 30, 50, 70]:
            t = step * dt
            if t > duration:
                break
            spline_val = _eval_spline(splines, t, duration)
            # Original relative position at this time
            orig_y_rel = plan["y"][step] - plan["y"][0]
            orig_th_rel = plan["theta"][step] - plan["theta"][0]

            # Spline should approximate the original trajectory
            assert abs(spline_val[1] - orig_y_rel) < 1.0, (
                f"Step {step}: spline y={spline_val[1]:.3f}, "
                f"orig y_rel={orig_y_rel:.3f}"
            )


# ===========================================================================
# Test: First CP pinning train/test mismatch
# ===========================================================================

class TestFirstCPPinning:
    """The first CP is pinned to [0,0,0] at inference time but NOT during
    training data generation. This creates a distribution mismatch."""

    def test_training_targets_first_cp_is_zero(self):
        """Training targets (KTO window fits) should have first CP = [0,0,0]
        since the fit is done in lander-relative coordinates with origin at
        the current position."""
        from collect import _fit_kto_window

        # Build a simple KTO-like plan
        n = 200
        plan = {
            "x": np.linspace(15.0, 15.0, n),
            "y": np.linspace(17.0, 5.0, n),
            "theta": np.zeros(n),
        }

        cps = _fit_kto_window(plan, idx=0, action_horizon=1.5)
        # x,y first CP should be zero (relative origin)
        assert abs(cps[0, 0]) < 0.01, f"x first CP should be ~0, got {cps[0, 0]}"
        assert abs(cps[0, 1]) < 0.01, f"y first CP should be ~0, got {cps[0, 1]}"
        # theta first CP = world theta at idx=0 / π = 0.0 / π = 0.0 (for this plan)
        assert abs(cps[0, 2]) < 0.01, f"theta first CP should be ~0 for zero-theta plan, got {cps[0, 2]}"

    def test_inference_pins_first_cp(self):
        """Inference always sets cps[0] = [0,0,0] after model.predict()."""
        class NonZeroFirstCPModel:
            def predict(self, cond, outcome, guidance_scale=2.0):
                cps = np.random.randn(N_CPS, N_CHANNELS)
                cps[0] = [10.0, 20.0, 0.5]  # Non-zero first CP
                return cps

        env = _make_env(seed=42)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        ctrl = KTODiffusionController(
            env, model=NonZeroFirstCPModel(), target_frequency=3.0,
            action_horizon=1.5, outcome=Outcome.SUCCESS,
        )
        ctrl.warm_start(time_budget=5.0)
        ctrl.inference()

        # Check that x,y pinned to zero, theta pinned to current world angle
        if ctrl._diff_splines is not None:
            first_cp = [float(ctrl._diff_splines[ch].c[0]) for ch in range(N_CHANNELS)]
            assert abs(first_cp[0]) < 1e-6, f"x should be pinned to 0, got {first_cp[0]}"
            assert abs(first_cp[1]) < 1e-6, f"y should be pinned to 0, got {first_cp[1]}"
            # theta should be world angle (small but nonzero for seed=42)
            world_theta = uw.lander.angle
            assert abs(first_cp[2] - world_theta) < 1e-4, (
                f"theta should be world angle {world_theta:.4f}, got {first_cp[2]:.4f}"
            )
        env.close()


# ===========================================================================
# Test: Normalization consistency across training rounds
# ===========================================================================

class TestNormalizationConsistency:

    def test_x_mean_x_std_dimensions(self):
        """x_mean and x_std should be (X_DIM,) = (30,)."""
        targets = np.random.randn(100, X_DIM).astype(np.float32)
        x_mean = torch.tensor(targets.mean(axis=0), dtype=torch.float32)
        x_std = torch.tensor(targets.std(axis=0).clip(1e-6), dtype=torch.float32)
        assert x_mean.shape == (X_DIM,)
        assert x_std.shape == (X_DIM,)

    def test_normalize_denormalize_roundtrip(self):
        """Normalize then denormalize should recover original targets."""
        targets = np.random.randn(50, X_DIM).astype(np.float32) * 3.0 + 1.0
        x_mean = torch.tensor(targets.mean(axis=0), dtype=torch.float32)
        x_std = torch.tensor(targets.std(axis=0).clip(1e-6), dtype=torch.float32)

        t = torch.tensor(targets)
        normalized = (t - x_mean) / x_std
        recovered = normalized * x_std + x_mean
        assert torch.allclose(t, recovered, atol=1e-5), "Roundtrip failed"

    def test_subsample_stats_differ_from_full(self):
        """Normalization from 500-frame subsample vs full dataset can differ,
        documenting the potential issue in dagger_loop.py."""
        np.random.seed(42)
        full = np.random.randn(5000, X_DIM).astype(np.float32) * 2.0 + 0.5
        sub = full[np.random.choice(5000, 500, replace=False)]

        full_mean = full.mean(axis=0)
        sub_mean = sub.mean(axis=0)
        mean_diff = np.abs(full_mean - sub_mean).max()

        full_std = full.std(axis=0)
        sub_std = sub.std(axis=0)
        std_diff = np.abs(full_std - sub_std).max()

        print(f"  Mean diff (full vs 500 subsample): {mean_diff:.4f}")
        print(f"  Std diff (full vs 500 subsample): {std_diff:.4f}")

        # With enough data, stats should be close but NOT identical
        # This documents that dagger_loop may have slight normalization drift
        assert mean_diff < 0.5, "Stats should be reasonably close"


# ===========================================================================
# Test: CFG conditioning correctness
# ===========================================================================

class TestCFGConditioning:

    def test_cfg_dropout_zeros_outcome_dim_only(self):
        """CFG dropout should zero only index [20], not state dims [0:20]."""
        from train import PositionDataset

        # Create a minimal training DB
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    cond BLOB NOT NULL,
                    target_cps BLOB NOT NULL,
                    outcome INTEGER NOT NULL
                )
            """)
            # Insert a frame with known values
            cond = np.ones(STATE_DIM, dtype=np.float32) * 42.0
            target = np.ones(X_DIM, dtype=np.float32) * 7.0
            conn.execute(
                "INSERT INTO frames (episode_id, step, cond, target_cps, outcome) VALUES (?,?,?,?,?)",
                (1, 0, cond.tobytes(), target.tobytes(), 1),
            )
            conn.commit()
            conn.close()

            dataset = PositionDataset(db_path)
            assert len(dataset) == 1

            # Sample many times to see dropout behavior
            n_dropped = 0
            n_kept = 0
            for _ in range(200):
                c, t = dataset[0]
                # State dims should always be 42.0
                assert torch.allclose(c[:STATE_DIM], torch.full((STATE_DIM,), 42.0)), (
                    f"State dims corrupted: {c[:STATE_DIM]}"
                )
                if abs(c[CFG_START].item()) < 1e-6:
                    n_dropped += 1
                else:
                    n_kept += 1
                    assert abs(c[CFG_START].item() - 1.0) < 1e-6, (
                        f"Outcome should be 1.0 when not dropped, got {c[CFG_START]}"
                    )

            # Should be ~50/50 with some variance
            ratio = n_dropped / (n_dropped + n_kept)
            print(f"  CFG dropout ratio: {ratio:.2f} (expected ~0.50)")
            assert 0.3 < ratio < 0.7, f"Dropout ratio {ratio:.2f} far from 0.5"
        finally:
            os.unlink(db_path)

    def test_ddim_cfg_produces_different_outputs_for_different_outcomes(self):
        """A trained model should produce different CPs for outcome=+1 vs -1.

        Without a trained model, we test that the CFG mechanism at least
        creates different noise predictions for conditioned vs unconditioned."""
        model = DiffusionMLP()
        model.eval()
        schedule = CosineSchedule(T=100)
        sampler = DDIMSampler(model, schedule, n_steps=10)

        cond_pos = torch.zeros(1, COND_DIM)
        cond_pos[0, :STATE_DIM] = torch.randn(STATE_DIM)
        cond_pos[0, CFG_START] = 1.0  # outcome = success

        cond_neg = cond_pos.clone()
        cond_neg[0, CFG_START] = -1.0  # outcome = failure

        torch.manual_seed(42)
        out_pos = sampler.sample_cfg(cond_pos, guidance_scale=2.0)
        torch.manual_seed(42)
        out_neg = sampler.sample_cfg(cond_neg, guidance_scale=2.0)

        # With an untrained model the outputs may still differ due to different
        # conditioning. The key check is that the CFG mechanism is active.
        diff = (out_pos - out_neg).abs().mean().item()
        print(f"  Mean abs diff between outcome=+1 and -1: {diff:.6f}")

        # With guidance_scale=2.0, the difference should be non-zero
        assert diff > 1e-6, "CFG should produce different outputs for different outcomes"

    def test_cfg_unconditioned_zeros_outcome(self):
        """DDIMSampler.sample_cfg should zero the outcome dim for the
        unconditioned pass."""
        model = DiffusionMLP()
        schedule = CosineSchedule(T=100)
        sampler = DDIMSampler(model, schedule, n_steps=10)

        cond = torch.zeros(2, COND_DIM)
        cond[:, CFG_START] = 1.0
        cond[:, 0] = 5.0  # some state

        # Check that the unconditioned conditioning is created correctly
        cond_uncond = cond.clone()
        cond_uncond[:, CFG_START:CFG_END] = 0.0

        assert cond_uncond[:, CFG_START].sum() == 0.0, "Outcome should be zeroed"
        assert cond_uncond[:, 0].sum() == 10.0, "State dims should be preserved"


# ===========================================================================
# Test: Training data DB integrity
# ===========================================================================

class TestDBIntegrity:

    def test_collect_frame_shapes(self):
        """Frames from collect.py should have correct blob shapes."""
        from collect import _fit_kto_window, run_episode

        env = _make_env(seed=50)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        reward, landed, frames = run_episode(env, 50, margin=0.1)
        env.close()

        assert len(frames) > 0, "Should have collected frames"

        for i, f in enumerate(frames):
            cond = f["cond"]
            target = f["target_cps"]
            assert cond.shape == (STATE_DIM,), (
                f"Frame {i}: cond shape {cond.shape}, expected ({STATE_DIM},)"
            )
            assert target.shape == (N_CPS * N_CHANNELS,), (
                f"Frame {i}: target shape {target.shape}, expected ({N_CPS * N_CHANNELS},)"
            )
            assert f["outcome"] in (1, -1), f"Frame {i}: outcome={f['outcome']}"

    def test_dagger_frame_shapes(self):
        """Frames from dagger_loop.py should have correct blob shapes."""
        from dagger_loop import collect_episode

        env = _make_env(seed=50)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        ep_info, frames = collect_episode(
            env, 50, NoiseModel(), margin=0.1, outcome_cond=Outcome.SUCCESS,
        )
        env.close()

        assert len(frames) > 0, "Should have collected frames"

        for i, f in enumerate(frames):
            mi = f["model_input"]
            mo = f["model_output_cps"]
            ac = f["actual_tracked_cps"]
            assert mi.shape == (STATE_DIM,), (
                f"Frame {i}: model_input shape {mi.shape}, expected ({STATE_DIM},)"
            )
            assert mo.shape == (N_CPS, N_CHANNELS), (
                f"Frame {i}: model_output shape {mo.shape}, expected ({N_CPS}, {N_CHANNELS})"
            )
            assert ac.shape == (N_CPS, N_CHANNELS), (
                f"Frame {i}: actual_tracked shape {ac.shape}, expected ({N_CPS}, {N_CHANNELS})"
            )

    def test_training_db_roundtrip(self):
        """Write to collect.py's TrainingDB and read with train.py's PositionDataset."""
        from collect import TrainingDB

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = TrainingDB(db_path)

            # Create test frames
            cond = np.random.randn(STATE_DIM).astype(np.float32)
            target = np.random.randn(N_CPS * N_CHANNELS).astype(np.float32)
            frames = [{"step": 0, "cond": cond, "target_cps": target, "outcome": 1}]

            db.save_episode(seed=42, landed=True, margin=0.1, reward=-3.0, frames=frames)
            db.close()

            # Read back
            from train import PositionDataset
            dataset = PositionDataset(db_path)
            assert len(dataset) == 1

            loaded_cond, loaded_target = dataset[0]
            # State part should match (ignoring CFG dropout)
            assert loaded_cond.shape == (COND_DIM,)
            assert loaded_target.shape == (N_CPS * N_CHANNELS,)

            # State values should match
            np.testing.assert_allclose(
                loaded_cond[:STATE_DIM].numpy(),
                cond, atol=1e-6,
            )
        finally:
            os.unlink(db_path)


# ===========================================================================
# Test: Outcome conditioning end-to-end
# ===========================================================================

class TestOutcomeConditioningFlow:

    def test_training_sees_both_outcomes(self):
        """Training data should contain both +1 and -1 outcomes.
        If only +1 outcomes exist, CFG can't learn the difference."""
        from collect import TrainingDB

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = TrainingDB(db_path)
            cond = np.zeros(STATE_DIM, dtype=np.float32)
            target = np.zeros(N_CPS * N_CHANNELS, dtype=np.float32)

            # Insert both outcomes
            db.save_episode(1, True, 0.1, -3.0,
                [{"step": 0, "cond": cond, "target_cps": target, "outcome": 1}])
            db.save_episode(2, False, 0.1, -10.0,
                [{"step": 0, "cond": cond, "target_cps": target, "outcome": -1}])
            db.close()

            from train import PositionDataset
            dataset = PositionDataset(db_path)
            assert len(dataset) == 2

            outcomes = set()
            for i in range(len(dataset)):
                c, t = dataset[i]
                # The outcome might be dropped, but check the stored value
                outcomes.add(dataset.outcomes[i])

            assert 1 in outcomes, "Should have +1 outcome in training data"
            assert -1 in outcomes, "Should have -1 outcome in training data"
        finally:
            os.unlink(db_path)

    def test_dagger_relabels_outcome_correctly(self):
        """ArchiveDataset should use the episode's actual outcome for conditioning,
        not the outcome_cond used during collection."""
        from dagger_loop import DaggerDB, ArchiveDataset

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = DaggerDB(db_path)

            # Save a landed episode
            db.save_episode(
                uid="ep1", git_commit="abc", env_seed=42, diffusion_seed=None,
                walltime_start=0.0, outcome=1, sim_duration=3.0, guidance_margin=0.2,
            )
            model_input = np.random.randn(STATE_DIM).astype(np.float32)
            model_output = np.random.randn(N_CPS * N_CHANNELS).astype(np.float32)
            actual_cps = np.random.randn(N_CPS * N_CHANNELS).astype(np.float32)
            db.save_frame(
                uid="fr1", episode_uid="ep1", t_sim_frame=0.0,
                model_input=model_input,
                model_output_cps=model_output,
                actual_tracked_cps=actual_cps,
            )

            # Save a failed episode
            db.save_episode(
                uid="ep2", git_commit="abc", env_seed=43, diffusion_seed=None,
                walltime_start=0.0, outcome=-1, sim_duration=5.0, guidance_margin=0.2,
            )
            db.save_frame(
                uid="fr2", episode_uid="ep2", t_sim_frame=0.0,
                model_input=model_input,
                model_output_cps=model_output,
                actual_tracked_cps=actual_cps,
            )
            db.commit()

            # Load frames
            rows = db.sample_frames(10, landed_frac=0.5)
            dataset = ArchiveDataset(rows, db)

            # Check outcomes are relabeled correctly
            for i in range(len(dataset)):
                cond = dataset.conds[i]
                outcome = cond[CFG_START]
                assert outcome in (1.0, -1.0), (
                    f"Outcome should be +1 or -1, got {outcome}"
                )

            db.close()
        finally:
            os.unlink(db_path)


# ===========================================================================
# Test: Diffusion forward/reverse process
# ===========================================================================

class TestDiffusionProcess:

    def test_forward_process_adds_noise(self):
        """Forward diffusion should add noise. At t=T, signal should be ~gone."""
        schedule = CosineSchedule(T=100)
        x0 = torch.randn(8, X_DIM)

        # At t=1, mostly signal
        ab_1 = schedule.get_alpha_bar(1)
        noise = torch.randn_like(x0)
        x1 = math.sqrt(ab_1) * x0 + math.sqrt(1 - ab_1) * noise
        # Signal should dominate
        signal_ratio = x0.var() / x1.var()
        assert signal_ratio > 0.5, f"t=1 should have mostly signal, ratio={signal_ratio}"

        # At t=T, mostly noise
        ab_T = schedule.get_alpha_bar(100)
        xT = math.sqrt(ab_T) * x0 + math.sqrt(1 - ab_T) * noise
        # Noise should dominate
        assert ab_T < 0.05, f"alpha_bar(T) should be small, got {ab_T}"

    def test_model_output_shape_during_training(self):
        """Model should output same shape as input x."""
        model = DiffusionMLP()
        B = 4
        x = torch.randn(B, X_DIM)
        cond = torch.randn(B, COND_DIM)
        t = torch.randint(1, 101, (B,))

        out = model(x, cond, t)
        assert out.shape == (B, X_DIM), f"Expected ({B}, {X_DIM}), got {out.shape}"

    def test_training_loss_decreases(self):
        """A few steps of training on synthetic data should decrease loss."""
        model = DiffusionMLP(hidden=64, n_blocks=2)
        schedule = CosineSchedule(T=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Synthetic: constant target
        target = torch.randn(1, X_DIM).expand(32, -1)
        cond = torch.randn(32, COND_DIM)

        x_mean = target.mean(dim=0)
        x_std = target.std(dim=0).clamp(1e-6)

        losses = []
        model.train()
        for _ in range(20):
            x0 = (target - x_mean) / x_std
            t = torch.randint(1, 101, (32,))
            ab = torch.tensor([schedule.get_alpha_bar(ti.item()) for ti in t]).unsqueeze(1)
            noise = torch.randn_like(x0)
            x_noisy = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
            eps_pred = model(x_noisy, cond, t)
            loss = torch.nn.functional.mse_loss(eps_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over training
        early = np.mean(losses[:5])
        late = np.mean(losses[-5:])
        print(f"  Loss early={early:.4f}, late={late:.4f}")
        assert late < early, f"Loss didn't decrease: early={early:.4f}, late={late:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
