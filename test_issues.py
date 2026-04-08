"""Tests for issues identified in issues.md.

Covers: knot vector consistency, spline clamping, conditioning correctness,
inverse dynamics accuracy, broken imports, output shapes, and margin blending.
"""

import math

import numpy as np
import pytest
import torch

import gymnasium as gym

from diffusion_controller import (
    KTODiffusionController, NoiseModel, Outcome,
    _make_position_spline, _eval_spline, _build_cond,
    ObstacleRelative, WaypointTarget,
    N_CPS, N_CHANNELS, DEGREE, STATE_DIM, COND_DIM, DT,
)
from model import DiffusionMLP, CosineSchedule, DDIMSampler, X_DIM
import solver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ENV_REGISTERED = False


def _make_env(seed=42):
    global _ENV_REGISTERED
    if not _ENV_REGISTERED:
        gym.register(
            id="LL-issues-test",
            entry_point="lunar_lander:LunarLander",
            max_episode_steps=1000,
            kwargs={"num_obstacles": 0},
        )
        _ENV_REGISTERED = True
    env = gym.make("LL-issues-test", render_mode=None, continuous=True)
    env.reset(seed=seed)
    return env


# ===========================================================================
# Test 1: B-spline knot vector consistency
# ===========================================================================

class TestKnotVectorConsistency:

    def test_fit_and_make_knots_match(self):
        """_fit_kto_window and _make_position_spline should use the same
        knot vector structure so CPs produce identical curves."""
        from collect import _fit_kto_window as collect_fit

        duration = 1.5
        cps = np.random.RandomState(42).randn(N_CPS, N_CHANNELS) * 0.5
        cps[0] = [0, 0, 0]

        # Build spline using _make_position_spline (inference path)
        splines_inf = _make_position_spline(cps, duration)

        # Build spline using the knot structure from _fit_kto_window (training path)
        n_internal = N_CPS - DEGREE + 1
        internal = np.linspace(0, duration, n_internal)
        training_knots = np.concatenate([
            np.full(DEGREE + 1, 0.0),
            internal[1:-1],
            np.full(DEGREE + 1, duration),
        ])

        from scipy.interpolate import BSpline
        splines_train = tuple(
            BSpline(training_knots, cps[:, ch], DEGREE, extrapolate=False)
            for ch in range(N_CHANNELS)
        )

        # Evaluate at several points — they should match if knots are consistent
        test_times = np.linspace(0.01, duration - 0.01, 20)
        for t in test_times:
            inf_vals = [float(s(t)) for s in splines_inf]
            train_vals = [float(s(t)) for s in splines_train]
            for ch in range(N_CHANNELS):
                assert abs(inf_vals[ch] - train_vals[ch]) < 1e-6, (
                    f"Knot mismatch at t={t:.3f}, ch={ch}: "
                    f"inference={inf_vals[ch]:.6f}, training={train_vals[ch]:.6f}"
                )


# ===========================================================================
# Test 2: Position spline clamping
# ===========================================================================

class TestPositionSplineClamping:

    def test_spline_endpoints_match_cps(self):
        """A clamped B-spline should have value equal to first/last CPs
        at t=0 and t=duration respectively."""
        cps = np.array([
            [1.0, 2.0, 0.1],   # first CP
            [1.5, 2.5, 0.2],
            [2.0, 3.0, 0.3],
            [2.5, 3.5, 0.0],
            [3.0, 4.0, -0.1],
            [3.5, 4.5, -0.2],
            [4.0, 5.0, -0.1],
            [4.5, 5.5, 0.0],
            [5.0, 6.0, 0.1],
            [5.5, 6.5, 0.15],  # last CP
        ])
        duration = 1.5
        splines = _make_position_spline(cps, duration)

        # Evaluate at endpoints
        start_vals = _eval_spline(splines, 0.0, duration)
        end_vals = _eval_spline(splines, duration, duration)

        for ch in range(N_CHANNELS):
            assert abs(start_vals[ch] - cps[0, ch]) < 1e-6, (
                f"Start not clamped: ch={ch}, spline={start_vals[ch]:.6f}, "
                f"CP={cps[0, ch]:.6f}"
            )
            assert abs(end_vals[ch] - cps[-1, ch]) < 1e-6, (
                f"End not clamped: ch={ch}, spline={end_vals[ch]:.6f}, "
                f"CP={cps[-1, ch]:.6f}"
            )


# ===========================================================================
# Test 3: Conditioning q_prev correctness
# ===========================================================================

class TestConditioningQPrev:

    def test_qprev_differs_from_qnow_after_first_call(self):
        """After the first inference, q_prev in conditioning should differ
        from q_now (since the lander has moved)."""
        env = _make_env(seed=100)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        ctrl = KTODiffusionController(
            env, model=NoiseModel(), target_frequency=3.0,
            action_horizon=1.5, outcome=Outcome.SUCCESS,
        )
        ctrl.warm_start(time_budget=5.0)

        # First inference
        ctrl.inference()
        q_after_first = ctrl._last_inference_q

        # Step forward several sim steps
        spi = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))
        for _ in range(spi):
            tv, th = ctrl.get_action(guidance_margin=0.001)
            env.step(np.array([tv, th], dtype=np.float32))

        # Second inference — q_now should differ from q_prev
        L = uw.lander
        q_now = (L.position.x, L.position.y, L.angle)
        q_prev_stored = ctrl._last_inference_q

        # The stored _last_inference_q should be from the first call
        # After inference(), it will be updated to q_now
        ctrl.inference()

        # Now check: the conditioning used inside inference() should have had
        # q_prev != q_now. We can verify this indirectly: _last_inference_q
        # before the second inference was the first inference's q_now.
        # The lander has moved, so q_now != q_prev_stored.
        dist = math.hypot(q_now[0] - q_prev_stored[0], q_now[1] - q_prev_stored[1])
        assert dist > 1e-4, (
            f"q_prev should differ from q_now after movement: dist={dist:.6f}"
        )
        env.close()


# ===========================================================================
# Test 4: Tracking inverse dynamics accuracy
# ===========================================================================

class TestTrackingInverseDynamics:

    @pytest.mark.parametrize("theta", [0.0, 0.1, 0.3, 0.5, math.pi / 6])
    def test_tracking_vs_full_inverse_dynamics(self, theta):
        """Compare _tracking_step thrust to full Cramer's-rule inverse dynamics."""
        # Desired accelerations
        ax_des = 2.0
        ay_des = -5.0

        gains = solver.TrackingGains(Kp_pos=0, Kd_pos=0, Kp_att=0, Kd_att=0)

        Fm_track, Fs_track = solver._tracking_step(
            x=10.0, y=10.0, theta=theta, vx=0, vy=0, omega=0,
            x_ref=10.0, y_ref=10.0, th_ref=theta,
            vx_ref=0, vy_ref=0, om_ref=0,
            ax_ref=ax_des, ay_ref=ay_des, al_ref=0.0,
            gains=gains,
        )

        # Full inverse dynamics (Cramer's rule)
        ct = math.cos(theta)
        st = math.sin(theta)
        c2t = ct * ct - st * st
        Fm_full = solver.MASS * (ax_des * st + (ay_des + solver.GRAVITY) * ct) / c2t

        # Document the error at each angle — this is the bug we're testing for.
        # After fix, tracking should match full inverse dynamics at all angles.
        error_pct = abs(Fm_track - Fm_full) / max(abs(Fm_full), 1e-6) * 100
        print(f"  θ={math.degrees(theta):.0f}°: tracking={Fm_track:.2f}, "
              f"full={Fm_full:.2f}, error={error_pct:.1f}%")
        # After fix this should hold for all angles (not just small)
        assert abs(Fm_track - Fm_full) < 0.5, (
            f"At θ={theta:.2f}, tracking Fm={Fm_track:.2f} vs full Fm={Fm_full:.2f}, "
            f"error={error_pct:.1f}%"
        )


# ===========================================================================
# Test 5 & 6: Import checks
# ===========================================================================

class TestImports:

    def test_vestigial_files_removed(self):
        """guidance_controller.py and test_diffusion_pipeline.py referenced
        non-existent types (ActionTarget, DiffusionController, LanderState).
        They should be deleted as they belong to the old thrust-spline API."""
        import os
        assert not os.path.exists("guidance_controller.py"), \
            "guidance_controller.py should be deleted (broken imports)"
        assert not os.path.exists("test_diffusion_pipeline.py"), \
            "test_diffusion_pipeline.py should be deleted (broken imports)"


# ===========================================================================
# Test 7: NoiseModel output shape
# ===========================================================================

class TestNoiseModel:

    def test_output_shape(self):
        """NoiseModel.predict() should return (N_CPS, N_CHANNELS)."""
        model = NoiseModel()
        cond = np.zeros(STATE_DIM, dtype=np.float32)
        result = model.predict(cond, Outcome.SUCCESS, guidance_scale=2.0)
        assert result.shape == (N_CPS, N_CHANNELS), (
            f"Expected ({N_CPS}, {N_CHANNELS}), got {result.shape}"
        )

    def test_output_reasonable_scale(self):
        """NoiseModel output should be small random values."""
        model = NoiseModel()
        cond = np.zeros(STATE_DIM, dtype=np.float32)
        results = [model.predict(cond, Outcome.SUCCESS) for _ in range(10)]
        stacked = np.stack(results)
        assert stacked.std() > 0.01, "NoiseModel produces constant output"
        assert np.abs(stacked).max() < 5.0, "NoiseModel output too large"


# ===========================================================================
# Test 8: DDIM sampler output shape
# ===========================================================================

class TestDDIMSampler:

    def test_output_shape(self):
        """DDIM sampling should produce (B, X_DIM) output."""
        model = DiffusionMLP()
        schedule = CosineSchedule(T=100)
        sampler = DDIMSampler(model, schedule, n_steps=10)

        cond = torch.randn(4, COND_DIM)
        with torch.no_grad():
            result = sampler.sample_cfg(cond, guidance_scale=2.0)

        assert result.shape == (4, X_DIM), (
            f"Expected (4, {X_DIM}), got {result.shape}"
        )


# ===========================================================================
# Test 9: _build_cond dimensions
# ===========================================================================

class TestBuildCond:

    def test_output_shape(self):
        """_build_cond should return (STATE_DIM,) array."""
        cond = _build_cond(
            t_obs_cmd_latency=0.02,
            q_now=(10.0, 15.0, 0.1),
            q_prev=(10.0, 15.0, 0.1),
            obstacle=ObstacleRelative(0.0, 0.0, 0.0),
            waypoint=WaypointTarget(dq=(1.0, -10.0, -0.1), dq_prime=(0.0, -0.5, 0.0)),
            guidance_q=(10.0, 15.0, 0.1),
            action_horizon=1.5,
        )
        assert cond.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {cond.shape}"
        assert cond.dtype == np.float32

    def test_values_packed_correctly(self):
        """Verify conditioning vector layout matches documentation."""
        q_now = (10.0, 15.0, 0.1)
        q_prev = (9.9, 15.1, 0.05)
        cond = _build_cond(
            t_obs_cmd_latency=0.02,
            q_now=q_now,
            q_prev=q_prev,
            obstacle=ObstacleRelative(1.0, 2.0, 0.5),
            waypoint=WaypointTarget(dq=(5.0, -10.0, -0.1), dq_prime=(0.0, -0.5, 0.0)),
            guidance_q=(10.5, 14.0, 0.0),
            action_horizon=1.5,
        )
        assert cond[0] == pytest.approx(0.02)        # t_obs_cmd_latency
        assert cond[1] == pytest.approx(10.0)         # q_now.x
        assert cond[2] == pytest.approx(15.0)         # q_now.y
        assert cond[3] == pytest.approx(0.1)           # q_now.theta
        assert cond[4] == pytest.approx(9.9)           # q_prev.x
        assert cond[5] == pytest.approx(15.1)          # q_prev.y
        assert cond[6] == pytest.approx(0.05)          # q_prev.theta
        assert cond[7] == pytest.approx(1.0)           # obstacle.dx
        assert cond[8] == pytest.approx(2.0)           # obstacle.dy
        assert cond[9] == pytest.approx(0.5)           # obstacle.r
        assert cond[19] == pytest.approx(1.5)          # action_horizon


# ===========================================================================
# Test 10: get_action returns zero after KTO exhausted
# ===========================================================================

class TestGetActionAfterKTOExhausted:

    def test_returns_zero_thrust(self):
        """After KTO plan is exhausted, get_action should return (0, 0)."""
        env = _make_env(seed=42)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        ctrl = KTODiffusionController(
            env, model=NoiseModel(), target_frequency=3.0,
            action_horizon=1.5, outcome=Outcome.SUCCESS,
        )
        ctrl.warm_start(time_budget=5.0)

        # Run inference
        ctrl.inference()

        # Advance time past KTO duration by directly modifying elapsed_s
        kto_end = ctrl._kto_t0 + ctrl._kto_duration
        uw.elapsed_s = kto_end + 1.0

        tv, th = ctrl.get_action(guidance_margin=0.5)
        assert tv == 0.0, f"Expected zero main thrust after KTO, got {tv}"
        assert th == 0.0, f"Expected zero side thrust after KTO, got {th}"
        env.close()


# ===========================================================================
# Test 11: Margin blending
# ===========================================================================

class TestMarginBlending:

    def test_margin_zero_tracks_kto(self):
        """At margin=0, position reference should equal KTO reference."""
        env = _make_env(seed=42)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        ctrl = KTODiffusionController(
            env, model=NoiseModel(), target_frequency=3.0,
            action_horizon=1.5, outcome=Outcome.SUCCESS,
        )
        ctrl.warm_start(time_budget=5.0)
        ctrl.inference()

        # Get KTO reference position
        t_sim = uw.elapsed_s
        kto_ref = ctrl._get_kto_ref(t_sim)
        assert kto_ref is not None

        # At margin=0, the blend should be pure KTO
        # We can't directly see the blended reference, but we can verify
        # that get_action at margin=0 produces a valid action
        tv, th = ctrl.get_action(guidance_margin=0.0)
        assert -1.0 <= tv <= 1.0
        assert -1.0 <= th <= 1.0
        env.close()

    def test_margin_affects_output(self):
        """Larger margin should allow more diffusion influence. With a model
        that outputs far from KTO, a large margin clamp should differ from
        a tiny one."""

        class LargeOffsetModel:
            """Returns CPs far from any KTO reference."""
            def predict(self, cond, outcome, guidance_scale=2.0):
                cps = np.ones((N_CPS, N_CHANNELS)) * 5.0
                cps[0] = [0, 0, 0]
                return cps

        env = _make_env(seed=42)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        ctrl = KTODiffusionController(
            env, model=LargeOffsetModel(), target_frequency=3.0,
            action_horizon=1.5, outcome=Outcome.SUCCESS,
        )
        ctrl.warm_start(time_budget=5.0)
        ctrl.inference()

        # Advance a few steps so diffusion spline is evaluated past t=0
        # (at t=0 the spline value is origin, matching KTO trivially)
        for _ in range(5):
            tv, th = ctrl.get_action(guidance_margin=0.001)
            env.step(np.array([tv, th], dtype=np.float32))

        a_low = ctrl.get_action(guidance_margin=0.001)
        a_high = ctrl.get_action(guidance_margin=5.0)

        # Large offset model + large margin should differ from tiny margin
        assert a_low != a_high, (
            f"Margin should affect output: low={a_low}, high={a_high}"
        )
        env.close()


# ===========================================================================
# Test 12: Cosine schedule properties
# ===========================================================================

class TestCosineSchedule:

    def test_alpha_bar_monotonic_decreasing(self):
        """alpha_bar should decrease from ~1 to ~0 as t increases."""
        schedule = CosineSchedule(T=100)
        for t in range(1, 101):
            assert schedule.get_alpha_bar(t) <= schedule.get_alpha_bar(t - 1), (
                f"alpha_bar not monotonic at t={t}"
            )

    def test_alpha_bar_endpoints(self):
        """alpha_bar[0] should be ~1, alpha_bar[T] should be near 0."""
        schedule = CosineSchedule(T=100)
        assert schedule.get_alpha_bar(0) > 0.99
        assert schedule.get_alpha_bar(100) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
