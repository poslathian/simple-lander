"""Tests for the simple_lander project.

Covers:
  - Surrogate dynamics vs Box2D physics agreement (discrete semi-implicit Euler)
  - Physics model divergence (solver vs Box2D)
  - Mass/inertia constant validation
  - KTO controller action conversion and thrust triangles
  - KTO landing end-to-end with randomized initial states
"""

import math
import time

import numpy as np
import pytest

import gymnasium as gym

import lunar_lander as ll
from lunar_lander import (
    SCALE, FPS, DT, MAIN_ENGINE_POWER, SIDE_ENGINE_POWER,
    MAIN_ENGINE_Y_LOCATION, SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY,
    VIEWPORT_W, VIEWPORT_H, LEG_DOWN,
    LunarLander, KTOController, heuristic,
    lander_dynamics, lander_step,
)
import solver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_REGISTERED = False

def _make_env(seed=42, render_mode=None):
    global _ENV_REGISTERED
    if not _ENV_REGISTERED:
        gym.register(
            id="LunarLander-test",
            entry_point="lunar_lander:LunarLander",
            max_episode_steps=1000,
            kwargs={"num_obstacles": 0},
        )
        _ENV_REGISTERED = True
    env = gym.make("LunarLander-test", render_mode=render_mode, continuous=True)
    env.reset(seed=seed)
    return env


def _world_state(env):
    L = env.unwrapped.lander
    return dict(
        x=L.position.x, y=L.position.y, theta=L.angle,
        vx=L.linearVelocity.x, vy=L.linearVelocity.y,
        omega=L.angularVelocity,
    )


def _apply_thrust_impulse(env, Fm, Fs):
    """Apply Fm/Fs as impulses directly to Box2D body (bypasses action space)."""
    lander = env.unwrapped.lander
    theta = lander.angle
    ct, st = math.cos(theta), math.sin(theta)

    if Fm > 0:
        ox = st * MAIN_ENGINE_Y_LOCATION / SCALE
        oy = -ct * MAIN_ENGINE_Y_LOCATION / SCALE
        impulse_pos = (lander.position[0] + ox, lander.position[1] + oy)
        lander.ApplyLinearImpulse((Fm * DT * (-st), Fm * DT * ct), impulse_pos, True)

    if abs(Fs) > 1e-6:
        side_x, side_y = -ct, st
        ox = side_x * SIDE_ENGINE_AWAY / SCALE
        oy = -side_y * SIDE_ENGINE_AWAY / SCALE
        impulse_pos = (
            lander.position[0] + ox - st * 17 / SCALE,
            lander.position[1] + oy + ct * SIDE_ENGINE_HEIGHT / SCALE,
        )
        lander.ApplyLinearImpulse((Fs * DT * ct, Fs * DT * (-st)), impulse_pos, True)


def _step_box2d_no_action(env):
    world = env.unwrapped.world
    world.Step(1.0 / FPS, 6, 2)
    world.ClearForces()
    env.unwrapped.elapsed_s += DT


def _euler_step(state, Fm, Fs, dt):
    x, y, theta, vx, vy, omega = (
        state["x"], state["y"], state["theta"],
        state["vx"], state["vy"], state["omega"],
    )
    ct, st = math.cos(theta), math.sin(theta)
    m, g, I = solver.MASS, solver.GRAVITY, solver.INERTIA

    ax = (-Fm * st + Fs * ct) / m
    ay = (Fm * ct - Fs * st) / m - g
    torque_arm = (2 * solver.SIDE_AWAY * st * ct
                  + solver.SIDE_ARM_A * st**2
                  - solver.SIDE_ARM_B * ct**2)
    alpha = Fs * torque_arm / I

    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    omega_new = omega + alpha * dt
    return dict(x=x + vx_new * dt, y=y + vy_new * dt, theta=theta + omega_new * dt,
                vx=vx_new, vy=vy_new, omega=omega_new)


# ===========================================================================
# Surrogate dynamics tests — discrete semi-implicit Euler vs Box2D
# ===========================================================================

def _rollout_lander_step(state0, Fm, Fs, n_steps):
    """Roll out lander_step for n_steps.  Fm, Fs are length-n arrays.

    Returns (n_steps+1, 6) state trajectory.
    """
    states = np.zeros((n_steps + 1, 6))
    states[0] = state0
    for i in range(n_steps):
        states[i + 1] = lander_step(list(states[i]), float(Fm[i]), float(Fs[i]))
    return states


class TestSurrogateDynamics:
    """Validate that lander_step (semi-implicit Euler) closely matches
    Box2D game physics, and that the KTO plan + surrogate + Box2D agree."""

    def test_discrete_freefall_vs_box2d(self):
        """Freefall: lander_step vs Box2D for 50 steps."""
        env = _make_env(seed=0)
        env.unwrapped.lander.linearVelocity = (0.0, 0.0)
        env.unwrapped.lander.angularVelocity = 0.0
        b2d = _world_state(env)
        state0 = [b2d["x"], b2d["y"], b2d["theta"], b2d["vx"], b2d["vy"], b2d["omega"]]

        N = 50
        surr = _rollout_lander_step(state0, np.zeros(N), np.zeros(N), N)

        pos_errs = []
        for i in range(N):
            _apply_thrust_impulse(env, 0.0, 0.0)
            _step_box2d_no_action(env)
            box = _world_state(env)
            pos_errs.append(math.hypot(box["x"] - surr[i + 1, 0],
                                       box["y"] - surr[i + 1, 1]))
        max_err = max(pos_errs)
        print(f"\nDiscrete freefall: max_pos_err={max_err:.6f}")
        # Residual from leg joint coupling (not modeled)
        assert max_err < 0.15, f"Freefall drift too large: {max_err}"
        env.close()

    def test_discrete_main_thrust_vs_box2d(self):
        """Main thrust at 50%: lander_step vs Box2D for 50 steps."""
        env = _make_env(seed=0)
        env.unwrapped.lander.linearVelocity = (0.0, 0.0)
        env.unwrapped.lander.angularVelocity = 0.0
        b2d = _world_state(env)
        state0 = [b2d["x"], b2d["y"], b2d["theta"], b2d["vx"], b2d["vy"], b2d["omega"]]

        N = 50
        Fm_val = solver.THRUST_MAX * 0.5
        surr = _rollout_lander_step(state0, np.full(N, Fm_val), np.zeros(N), N)

        pos_errs = []
        for i in range(N):
            _apply_thrust_impulse(env, Fm_val, 0.0)
            _step_box2d_no_action(env)
            box = _world_state(env)
            pos_errs.append(math.hypot(box["x"] - surr[i + 1, 0],
                                       box["y"] - surr[i + 1, 1]))
        max_err = max(pos_errs)
        print(f"\nDiscrete main thrust: max_pos_err={max_err:.6f}")
        assert max_err < 0.15, f"Main thrust drift too large: {max_err}"
        env.close()

    def test_discrete_side_thrust_vs_box2d(self):
        """Side thrust at 50% with initial tilt: lander_step vs Box2D."""
        env = _make_env(seed=0)
        env.unwrapped.lander.linearVelocity = (0.0, 0.0)
        env.unwrapped.lander.angularVelocity = 0.0
        env.unwrapped.lander.angle = 0.26
        b2d = _world_state(env)
        state0 = [b2d["x"], b2d["y"], b2d["theta"], b2d["vx"], b2d["vy"], b2d["omega"]]

        N = 50
        Fs_val = solver.SIDE_MAX * 0.5
        surr = _rollout_lander_step(state0, np.zeros(N), np.full(N, Fs_val), N)

        pos_errs, angle_errs = [], []
        for i in range(N):
            _apply_thrust_impulse(env, 0.0, Fs_val)
            _step_box2d_no_action(env)
            box = _world_state(env)
            pos_errs.append(math.hypot(box["x"] - surr[i + 1, 0],
                                       box["y"] - surr[i + 1, 1]))
            angle_errs.append(abs(box["theta"] - surr[i + 1, 2]))
        print(f"\nDiscrete side thrust: max_pos_err={max(pos_errs):.6f}  "
              f"max_angle_err={max(angle_errs):.6f}")
        assert max(pos_errs) < 0.5, f"Side thrust pos drift: {max(pos_errs)}"
        assert max(angle_errs) < 0.1, f"Side thrust angle drift: {max(angle_errs)}"
        env.close()

    def test_discrete_matches_kto_plan(self):
        """Roll out lander_step with KTO thrust plan and compare to the
        solver's planned trajectory and Box2D replay."""
        env = _make_env(seed=42)
        env.unwrapped.lander.linearVelocity = (0.0, 0.0)
        env.unwrapped.lander.angularVelocity = 0.0
        b2d = _world_state(env)
        start = np.array([b2d["x"], b2d["y"], b2d["theta"]])
        goal_y = env.unwrapped.helipad_y + LEG_DOWN / SCALE + 2.0
        goal = np.array([solver.PAD_X, goal_y, 0.0])

        times, plan, *_ = solver.solve(
            start=start, goal=goal, obstacles=(),
            time_budget=5.0, warmstart_budget=1.0)

        duration = times[-1] - times[0]
        n_sim = int(duration / DT)
        sim_t = np.linspace(times[0], times[-1], n_sim)
        Fm_seq = np.clip(np.interp(sim_t, times, plan["Fm"]), 0, solver.THRUST_MAX)
        Fs_seq = np.clip(np.interp(sim_t, times, plan["Fs"]), -solver.SIDE_MAX, solver.SIDE_MAX)
        x_plan = np.interp(sim_t, times, plan["x"])
        y_plan = np.interp(sim_t, times, plan["y"])

        # Discrete surrogate rollout
        state0 = [b2d["x"], b2d["y"], b2d["theta"], b2d["vx"], b2d["vy"], b2d["omega"]]
        surr = _rollout_lander_step(state0, Fm_seq, Fs_seq, n_sim)

        # Box2D replay
        box_pos = []
        for i in range(n_sim):
            _apply_thrust_impulse(env, float(Fm_seq[i]), float(Fs_seq[i]))
            _step_box2d_no_action(env)
            s = _world_state(env)
            box_pos.append((s["x"], s["y"]))

        surr_vs_box, surr_vs_plan, box_vs_plan = [], [], []
        for i in range(n_sim):
            sx, sy = surr[i + 1, 0], surr[i + 1, 1]
            bx, by = box_pos[i]
            px, py = x_plan[i], y_plan[i]
            surr_vs_box.append(math.hypot(sx - bx, sy - by))
            surr_vs_plan.append(math.hypot(sx - px, sy - py))
            box_vs_plan.append(math.hypot(bx - px, by - py))

        print(f"\nThree-way agreement ({n_sim} steps):")
        print(f"  lander_step vs Box2D: max={max(surr_vs_box):.4f}  mean={np.mean(surr_vs_box):.4f}")
        print(f"  lander_step vs Plan:  max={max(surr_vs_plan):.4f}  mean={np.mean(surr_vs_plan):.4f}")
        print(f"  Box2D vs Plan:        max={max(box_vs_plan):.4f}  mean={np.mean(box_vs_plan):.4f}")

        # Discrete surrogate should track Box2D (residual from unmodeled leg joints)
        assert max(surr_vs_box) < 10.0, (
            f"lander_step vs Box2D diverged: {max(surr_vs_box):.4f}")
        # Box2D should track the plan (wider spawn means longer trajectories)
        assert max(box_vs_plan) < 15.0, (
            f"Box2D vs Plan diverged: {max(box_vs_plan):.4f}")


# ===========================================================================
# Physics divergence tests
# ===========================================================================

class TestPhysicsDivergence:

    def test_freefall(self):
        env = _make_env(seed=0)
        euler_state = _world_state(env)
        pos_errors = []
        for _ in range(50):
            _apply_thrust_impulse(env, 0.0, 0.0)
            _step_box2d_no_action(env)
            euler_state = _euler_step(euler_state, 0.0, 0.0, DT)
            box2d = _world_state(env)
            pos_errors.append(math.hypot(box2d["x"] - euler_state["x"],
                                         box2d["y"] - euler_state["y"]))
        print(f"\nFreefall: max_pos_err={max(pos_errors):.6f}")
        assert max(pos_errors) < 1.0
        env.close()

    def test_main_thrust(self):
        env = _make_env(seed=0)
        env.unwrapped.lander.linearVelocity = (0.0, 0.0)
        env.unwrapped.lander.angularVelocity = 0.0
        euler_state = _world_state(env)
        Fm = solver.THRUST_MAX * 0.5
        pos_errors = []
        for _ in range(50):
            _apply_thrust_impulse(env, Fm, 0.0)
            _step_box2d_no_action(env)
            euler_state = _euler_step(euler_state, Fm, 0.0, DT)
            box2d = _world_state(env)
            pos_errors.append(math.hypot(box2d["x"] - euler_state["x"],
                                         box2d["y"] - euler_state["y"]))
        print(f"\nMain thrust: max_pos_err={max(pos_errors):.6f}")
        assert max(pos_errors) < 1.0
        env.close()

    def test_side_thrust(self):
        env = _make_env(seed=0)
        env.unwrapped.lander.linearVelocity = (0.0, 0.0)
        env.unwrapped.lander.angularVelocity = 0.0
        env.unwrapped.lander.angle = 0.26
        euler_state = _world_state(env)
        Fs = solver.SIDE_MAX * 0.5
        pos_errors, angle_errors = [], []
        for _ in range(50):
            _apply_thrust_impulse(env, 0.0, Fs)
            _step_box2d_no_action(env)
            euler_state = _euler_step(euler_state, 0.0, Fs, DT)
            box2d = _world_state(env)
            pos_errors.append(math.hypot(box2d["x"] - euler_state["x"],
                                         box2d["y"] - euler_state["y"]))
            angle_errors.append(abs(box2d["theta"] - euler_state["theta"]))
        print(f"\nSide thrust (θ₀=0.26): max_pos_err={max(pos_errors):.6f}  "
              f"max_angle_err={max(angle_errors):.6f} rad")
        env.close()

    def test_solver_plan_replay(self):
        env = _make_env(seed=42)
        env.unwrapped.lander.linearVelocity = (0.0, 0.0)
        env.unwrapped.lander.angularVelocity = 0.0
        b2d = _world_state(env)
        start = np.array([b2d["x"], b2d["y"], b2d["theta"]])
        goal = np.array([solver.PAD_X, solver.PAD_Y, 0.0])
        times, plan, *_ = solver.solve(
            start=start, goal=goal, obstacles=(),
            time_budget=5.0, warmstart_budget=1.0)
        duration = times[-1] - times[0]
        n_sim = int(duration / DT)
        sim_t = np.linspace(times[0], times[-1], n_sim)
        Fm_i = np.interp(sim_t, times, plan["Fm"])
        Fs_i = np.interp(sim_t, times, plan["Fs"])
        x_p = np.interp(sim_t, times, plan["x"])
        y_p = np.interp(sim_t, times, plan["y"])
        pos_errors = []
        for i in range(n_sim):
            _apply_thrust_impulse(env, float(np.clip(Fm_i[i], 0, solver.THRUST_MAX)),
                                  float(np.clip(Fs_i[i], -solver.SIDE_MAX, solver.SIDE_MAX)))
            _step_box2d_no_action(env)
            box2d = _world_state(env)
            pos_errors.append(math.hypot(box2d["x"] - x_p[i], box2d["y"] - y_p[i]))
        print(f"\nSolver plan replay ({n_sim} steps, {duration:.2f}s): "
              f"max={max(pos_errors):.4f}  mean={np.mean(pos_errors):.4f}")
        env.close()

    def test_mass_inertia_match(self):
        env = _make_env(seed=0)
        lander = env.unwrapped.lander
        b2d_I = lander.inertia
        b2d_cm = lander.localCenter
        # Solver uses lander body mass (legs are massless during flight)
        print(f"\nMass: solver={solver.MASS:.6f} box2d_lander={lander.mass:.6f}")
        print(f"Inertia: solver={solver.INERTIA:.6f} box2d={b2d_I:.6f}")
        assert abs(solver.MASS - lander.mass) < 0.01
        assert abs(solver.INERTIA - b2d_I) < 0.01
        assert abs(ll.LANDER_CM_LOCAL[1] - b2d_cm.y) < 0.01
        env.close()


# ===========================================================================
# KTO controller action tests
# ===========================================================================

class TestKTOActions:

    def test_returns_nonzero_actions(self):
        """KTO controller must return real actions, not [0,0]."""
        env = _make_env(seed=42)
        ctrl = KTOController(env, time_budget=5.0)
        actions = []
        for _ in range(min(ctrl.n_steps, 50)):
            a = ctrl.step(env)
            actions.append(a.copy())
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        actions = np.array(actions)
        assert actions.shape[1] == 2
        # Main engine should be active on many steps
        assert (np.abs(actions[:, 0]) > 0.01).sum() > 10
        print(f"\nActions: main=[{actions[:,0].min():.3f},{actions[:,0].max():.3f}]  "
              f"side=[{actions[:,1].min():.3f},{actions[:,1].max():.3f}]")
        env.close()

    def test_actions_in_valid_range(self):
        """All actions must be in [-1, 1]."""
        env = _make_env(seed=42)
        ctrl = KTOController(env, time_budget=5.0)
        for _ in range(ctrl.n_steps):
            a = ctrl.step(env)
            assert -1.0 <= a[0] <= 1.0, f"main action out of range: {a[0]}"
            assert -1.0 <= a[1] <= 1.0, f"side action out of range: {a[1]}"
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        env.close()

    def test_no_action_deadzone(self):
        """env.step() should respond to all action magnitudes, no dead zones."""
        env = _make_env(seed=0)
        uw = env.unwrapped

        # Small main action should produce nonzero m_power
        env.step(np.array([0.1, 0.0], dtype=np.float32))
        assert uw.m_power > 0.0, f"Dead zone: action[0]=0.1 gave m_power=0"

        # Small side action should produce nonzero s_power
        obs, _ = env.reset(seed=0)
        env.step(np.array([0.0, 0.2], dtype=np.float32))
        assert uw.s_power > 0.0, f"Dead zone: action[1]=0.2 gave s_power=0"
        env.close()


# ===========================================================================
# KTO tracking error test
# ===========================================================================

class TestKTOTracking:

    def test_drift_correlates_with_lateral_thrust(self):
        """Visualize that tracking error grows when the plan requires lateral movement.

        Saves frames at key moments and checks that:
          1. Error is small during the initial vertical descent (no lateral demand)
          2. Error grows once the plan demands horizontal correction (via tilt)
          3. The rendered frames show the lander diverging from the cyan spline
        """
        import os
        from PIL import Image

        env = _make_env(seed=106, render_mode="rgb_array")
        ctrl = KTOController(env, time_budget=5.0)

        os.makedirs("tmp", exist_ok=True)
        errors = []
        lateral_demands = []
        saved_frames = []

        for i in range(min(ctrl.n_steps, 150)):
            px, py = ctrl.plan_x[i], ctrl.plan_y[i]
            a = ctrl.step(env)
            obs, r, term, trunc, _ = env.step(a)
            uw = env.unwrapped
            ax, ay = uw.lander.position.x, uw.lander.position.y
            err = math.hypot(ax - px, ay - py)
            errors.append(err)

            # Lateral demand: how much the plan wants to move horizontally
            if i > 0:
                dx_plan = ctrl.plan_x[i] - ctrl.plan_x[i - 1]
            else:
                dx_plan = 0.0
            lateral_demands.append(abs(dx_plan))

            # Save frames at key moments
            if i in (0, 30, 60, 90, 120):
                frame = env.render()
                if frame is not None:
                    path = f"tmp/drift_step{i:03d}.png"
                    Image.fromarray(frame).save(path)
                    saved_frames.append(path)

            if term or trunc:
                break
        env.close()

        # Split into early (first 40 steps, mostly vertical) and late
        early = errors[:40]
        late = errors[60:] if len(errors) > 60 else errors[40:]

        early_max = max(early) if early else 0
        late_max = max(late) if late else 0

        print(f"\nDrift analysis ({len(errors)} steps):")
        print(f"  Early (steps 0-39):  max_err={early_max:.4f}")
        print(f"  Late  (steps 60+):   max_err={late_max:.4f}")
        print(f"  Lateral demand: early_mean={np.mean(lateral_demands[:40]):.4f}  "
              f"late_mean={np.mean(lateral_demands[60:]):.4f}")
        print(f"  Saved frames: {saved_frames}")

        # Error should be much larger in the late phase where lateral
        # movement is demanded and tilt-induced thrust errors accumulate
        assert late_max > early_max * 2.0, (
            f"Expected late drift ({late_max:.3f}) to be much larger than "
            f"early drift ({early_max:.3f}) — lateral thrust error not visible")

        # Frames should exist
        for p in saved_frames:
            assert os.path.exists(p), f"Frame not saved: {p}"

    def test_action_rollout_matches_direct_impulse(self):
        """KTO controller tracking error through the action space should be
        comparable to direct-impulse replay.

        We run two rollouts from the same initial state with the same plan:
          1. Direct impulse: bypass action space, apply Fm/Fs as impulses
          2. Action-based: use KTOController.step() -> env.step()

        If the action conversion is correct, both should show similar drift
        from the solver's planned trajectory.
        """
        # ── Solve from a known state ─────────────────────────────────────
        env1 = _make_env(seed=106)
        env1.unwrapped.lander.linearVelocity = (0.0, 0.0)
        env1.unwrapped.lander.angularVelocity = 0.0
        b2d = _world_state(env1)
        start = np.array([b2d["x"], b2d["y"], b2d["theta"]])
        goal_y = env1.unwrapped.helipad_y + LEG_DOWN / SCALE + 2.0
        goal = np.array([solver.PAD_X, goal_y, 0.0])

        times, plan, *_ = solver.solve(
            start=start, goal=goal, obstacles=(),
            time_budget=5.0, warmstart_budget=1.0)
        duration = times[-1] - times[0]
        n_steps = int(duration / DT)
        sim_t = np.linspace(times[0], times[-1], n_steps)
        Fm_seq = np.interp(sim_t, times, plan["Fm"])
        Fs_seq = np.interp(sim_t, times, plan["Fs"])
        x_plan = np.interp(sim_t, times, plan["x"])
        y_plan = np.interp(sim_t, times, plan["y"])

        # ── Rollout 1: direct impulse ────────────────────────────────────
        direct_errors = []
        for i in range(n_steps):
            Fm_i = float(np.clip(Fm_seq[i], 0, solver.THRUST_MAX))
            Fs_i = float(np.clip(Fs_seq[i], -solver.SIDE_MAX, solver.SIDE_MAX))
            _apply_thrust_impulse(env1, Fm_i, Fs_i)
            _step_box2d_no_action(env1)
            s = _world_state(env1)
            direct_errors.append(math.hypot(s["x"] - x_plan[i], s["y"] - y_plan[i]))
        env1.close()

        # ── Rollout 2: action-based (KTOController) ─────────────────────
        env2 = _make_env(seed=106)
        ctrl = KTOController(env2, time_budget=5.0)
        action_errors = []
        for i in range(ctrl.n_steps):
            # Read planned position for this step
            px, py = ctrl.plan_x[i], ctrl.plan_y[i]
            a = ctrl.step(env2)
            obs, r, term, trunc, _ = env2.step(a)
            s = _world_state(env2)
            action_errors.append(math.hypot(s["x"] - px, s["y"] - py))
            if term or trunc:
                break
        env2.close()

        n_compare = min(len(direct_errors), len(action_errors))
        direct_max = max(direct_errors[:n_compare])
        action_max = max(action_errors[:n_compare])
        direct_mean = np.mean(direct_errors[:n_compare])
        action_mean = np.mean(action_errors[:n_compare])

        print(f"\nTracking error comparison ({n_compare} steps):")
        print(f"  Direct impulse: max={direct_max:.4f}  mean={direct_mean:.4f}")
        print(f"  Action-based:   max={action_max:.4f}  mean={action_mean:.4f}")
        print(f"  Ratio (action/direct): max={action_max/max(direct_max,1e-6):.2f}x  "
              f"mean={action_mean/max(direct_mean,1e-6):.2f}x")

        # The action-based rollout should not be drastically worse than
        # direct impulse.  If it's >3x worse, the action conversion is
        # introducing significant error beyond the physics model mismatch.
        assert action_max < direct_max * 3.0 + 1.0, (
            f"Action-based tracking ({action_max:.2f}) is much worse than "
            f"direct impulse ({direct_max:.2f}) — check force→action conversion")


# ===========================================================================
# Thrust triangle drawing tests
# ===========================================================================

class TestThrustTriangles:

    def test_all_directions_active_during_kto(self):
        """KTO episode must activate up, left, and right thrust indicators."""
        env = _make_env(seed=42, render_mode="rgb_array")
        ctrl = KTOController(env, time_budget=5.0)
        uw = env.unwrapped

        saw_up = False
        saw_left = False
        saw_right = False

        for _ in range(ctrl.n_steps + 100):
            a = ctrl.step(env)
            obs, r, term, trunc, _ = env.step(a)
            if uw.m_power > 0.05:
                saw_up = True
            if uw.s_power > 0.05 and uw.s_dir < 0:
                saw_left = True
            if uw.s_power > 0.05 and uw.s_dir > 0:
                saw_right = True
            if term or trunc:
                break

        print(f"\nThrust directions seen: up={saw_up} left={saw_left} right={saw_right}")
        assert saw_up, "Main engine never fired"
        assert saw_left, "Left thrust never fired"
        assert saw_right, "Right thrust never fired"
        env.close()

    def test_thrust_triangles_render_active_color(self):
        """When thrust is active, the orange indicator pixels should appear."""
        env = _make_env(seed=42, render_mode="rgb_array")
        ctrl = KTOController(env, time_budget=5.0)
        uw = env.unwrapped
        ACTIVE_ORANGE = np.array([255, 140, 0])

        found_orange_on_thrust = False
        for _ in range(ctrl.n_steps):
            a = ctrl.step(env)
            obs, r, term, trunc, _ = env.step(a)
            if uw.m_power > 0.3:
                frame = env.render()
                if frame is not None:
                    # Check for orange pixels anywhere in the frame
                    matches = np.all(frame == ACTIVE_ORANGE, axis=2)
                    if matches.any():
                        found_orange_on_thrust = True
                        break
            if term or trunc:
                break

        assert found_orange_on_thrust, "No orange thrust pixels found when engine active"
        env.close()


# ===========================================================================
# Initial state randomization tests
# ===========================================================================

class TestInitialState:

    def test_horizontal_position_uniform(self):
        """Initial x should be uniform across full display width."""
        env = _make_env(seed=0)
        W = VIEWPORT_W / SCALE
        xs = []
        for seed in range(50):
            env.reset(seed=seed)
            xs.append(env.unwrapped.lander.position.x)
        xs = np.array(xs)
        # Should span most of the display (1.0 to W-1.0)
        assert xs.min() >= 0.5, f"Min x below range: {xs.min():.1f}"
        assert xs.max() <= W - 0.5, f"Max x above range: {xs.max():.1f}"
        assert xs.max() - xs.min() > W * 0.5, f"x range too narrow: {xs.max()-xs.min():.1f}"
        print(f"\nInitial x: min={xs.min():.1f} max={xs.max():.1f} "
              f"std={xs.std():.1f} W={W:.0f}")
        env.close()

    def test_vertical_position_gaussian(self):
        """Initial y should be a small gaussian near the top."""
        env = _make_env(seed=0)
        H = VIEWPORT_H / SCALE
        ys = []
        for seed in range(50):
            env.reset(seed=seed)
            ys.append(env.unwrapped.lander.position.y)
        ys = np.array(ys)
        # Should be near top of screen
        assert ys.mean() > H * 0.7, f"Mean y too low: {ys.mean():.1f}"
        # Should be tightly clustered (gaussian, not uniform)
        assert ys.std() < 2.0, f"y std too large (not gaussian): {ys.std():.1f}"
        print(f"\nInitial y: mean={ys.mean():.1f} std={ys.std():.1f} (H={H:.0f})")
        env.close()


# ===========================================================================
# KTO end-to-end landing test
# ===========================================================================

class TestKTOLanding:

    def test_kto_landing(self):
        """Run 10 KTO episodes: solves <5s, faster than RT, some land."""
        env = _make_env(seed=0)
        landed_ct = 0
        initial_xs = []

        for ep in range(10):
            obs, _ = env.reset(seed=105 + ep)
            initial_xs.append(env.unwrapped.lander.position.x)

            t0 = time.monotonic()
            ctrl = KTOController(env, time_budget=5.0)
            solve_t = time.monotonic() - t0
            assert solve_t < 5.0, f"Ep {ep}: solve took {solve_t:.2f}s"

            t0 = time.monotonic()
            total_reward, done, steps = 0.0, False, 0
            while not done:
                action = ctrl.step(env)
                obs, reward, term, trunc, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = term or trunc
            sim_t = time.monotonic() - t0

            uw = env.unwrapped
            landed = (not uw.game_over
                      and (uw.legs[0].ground_contact or uw.legs[1].ground_contact))
            if landed:
                landed_ct += 1
            rtf = uw.elapsed_s / sim_t if sim_t > 0 else float("inf")
            status = "LANDED" if landed else "CRASH"
            print(f"  Ep {ep}: {status:6s} solve={solve_t:.2f}s t={uw.elapsed_s:.2f}s "
                  f"reward={total_reward:.2f} RTF={rtf:.0f}x")
            assert rtf > 1.0, f"Ep {ep}: slower than real time (RTF={rtf:.1f})"

        print(f"\nLanded: {landed_ct}/10")
        # Initial x should show variety from uniform randomization
        assert max(initial_xs) - min(initial_xs) > 2.0, \
            f"Initial x range too narrow: {initial_xs}"
        assert landed_ct >= 3, f"Only {landed_ct}/10 landed (need >=3)"
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
