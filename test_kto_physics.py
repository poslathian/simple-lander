"""Measure divergence between solver's continuous physics model and Box2D simulation.

The solver predicts (x, y, θ) trajectories using a rigid-body ODE model and computes
thrust profiles (Fm, Fs) via inverse dynamics.  This test applies those exact thrusts
to the Box2D env (bypassing action-space quantization) and measures how far the
simulated state drifts from the solver's prediction.

Sources of divergence:
  1. Integrator mismatch: solver uses continuous B-spline, Box2D uses semi-implicit Euler
  2. Leg/joint dynamics: solver treats the lander as a single rigid body; Box2D has
     revolute-jointed legs that absorb energy
"""

import math
import numpy as np
import pytest

import lunar_lander as ll
from lunar_lander import (
    SCALE, FPS, DT, MAIN_ENGINE_POWER, SIDE_ENGINE_POWER,
    MAIN_ENGINE_Y_LOCATION, SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY,
    VIEWPORT_W, VIEWPORT_H, LEG_DOWN,
)
import solver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(seed=42):
    """Create env with no rendering."""
    import gymnasium as gym
    gym.register(
        id="LunarLander-test",
        entry_point="lunar_lander:LunarLander",
        max_episode_steps=1000,
        kwargs={"num_obstacles": 0},
    )
    env = gym.make("LunarLander-test", render_mode=None, continuous=True)
    env.reset(seed=seed)
    return env


def _world_state(env):
    """Extract raw world-frame state from Box2D body."""
    L = env.unwrapped.lander
    return dict(
        x=L.position.x, y=L.position.y, theta=L.angle,
        vx=L.linearVelocity.x, vy=L.linearVelocity.y,
        omega=L.angularVelocity,
    )


def _apply_thrust_impulse(env, Fm, Fs):
    """Apply Fm/Fs as impulses to the Box2D lander, matching lunar_lander.py's
    force-application geometry (application points, directions).

    This bypasses the action-space clipping so we can test the physics model
    directly.
    """
    lander = env.unwrapped.lander
    theta = lander.angle
    ct, st = math.cos(theta), math.sin(theta)
    tip = (st, ct)
    side = (-ct, st)

    # Main engine impulse (same geometry as lunar_lander.py step())
    if Fm > 0:
        ox = tip[0] * MAIN_ENGINE_Y_LOCATION / SCALE
        oy = -tip[1] * MAIN_ENGINE_Y_LOCATION / SCALE
        impulse_pos = (lander.position[0] + ox, lander.position[1] + oy)
        # In step(): impulse = (-ox, -oy) * MAIN_ENGINE_POWER * m_power
        # We want total force = Fm in direction (-sinθ, cosθ),
        # so impulse = Fm * DT * (-sinθ, cosθ)
        fx = Fm * DT * (-st)
        fy = Fm * DT * ct
        lander.ApplyLinearImpulse((fx, fy), impulse_pos, True)

    # Side engine impulse (same geometry as lunar_lander.py step())
    if abs(Fs) > 1e-6:
        s_dir = 1.0 if Fs > 0 else -1.0
        ox = side[0] * SIDE_ENGINE_AWAY / SCALE
        oy = -side[1] * SIDE_ENGINE_AWAY / SCALE
        impulse_pos = (
            lander.position[0] + ox - tip[0] * 17 / SCALE,
            lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
        )
        # In step(): impulse = -s_dir * s_power * SIDE_ENGINE_POWER * side
        # We want force magnitude |Fs| in body-right direction
        # body-right in Box2D = (cosθ, -sinθ) (perpendicular to tip, CW 90°)
        # impulse = Fs * DT * (cosθ, -sinθ)
        fx = Fs * DT * ct
        fy = Fs * DT * (-st)
        lander.ApplyLinearImpulse((fx, fy), impulse_pos, True)


def _step_box2d_no_action(env):
    """Advance Box2D physics one step (no forces applied through action)."""
    world = env.unwrapped.world
    world.Step(1.0 / FPS, 6, 2)
    world.ClearForces()
    env.unwrapped.elapsed_s += DT


def _euler_step(state, Fm, Fs, dt):
    """One semi-implicit Euler step using the SOLVER's physics model.

    This replicates what the solver assumes (continuous rigid-body ODE),
    discretized the same way Box2D integrates (semi-implicit Euler).
    """
    x, y, theta, vx, vy, omega = (
        state["x"], state["y"], state["theta"],
        state["vx"], state["vy"], state["omega"],
    )
    ct, st = math.cos(theta), math.sin(theta)
    m = solver.MASS
    g = solver.GRAVITY
    I = solver.INERTIA

    # Solver's acceleration model (side engine direction: cosθ, -sinθ)
    ax = (-Fm * st + Fs * ct) / m
    ay = (Fm * ct - Fs * st) / m - g
    # Solver's torque model (full Box2D application-point cross product)
    torque_arm = (2 * solver.SIDE_AWAY * st * ct
                  + solver.SIDE_ARM_A * st**2
                  - solver.SIDE_ARM_B * ct**2)
    alpha = Fs * torque_arm / I

    # Semi-implicit Euler (velocity first, then position)
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    omega_new = omega + alpha * dt
    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    theta_new = theta + omega_new * dt

    return dict(x=x_new, y=y_new, theta=theta_new,
                vx=vx_new, vy=vy_new, omega=omega_new)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_freefall_divergence():
    """Zero thrust: measures integrator + leg-joint divergence only."""
    env = _make_env(seed=0)
    s0 = _world_state(env)
    euler_state = dict(s0)

    n_steps = 50  # 1 second of freefall
    pos_errors = []

    for _ in range(n_steps):
        _apply_thrust_impulse(env, 0.0, 0.0)
        _step_box2d_no_action(env)
        euler_state = _euler_step(euler_state, 0.0, 0.0, DT)

        box2d = _world_state(env)
        err = math.hypot(box2d["x"] - euler_state["x"],
                         box2d["y"] - euler_state["y"])
        pos_errors.append(err)

    max_err = max(pos_errors)
    final_err = pos_errors[-1]
    print(f"\nFreefall: max_pos_err={max_err:.6f}  final_pos_err={final_err:.6f}")
    # Expect small divergence (legs + numerical)
    assert max_err < 1.0, f"Freefall divergence too large: {max_err}"
    env.close()


def test_main_thrust_divergence():
    """Constant upward thrust: measures main-engine force model divergence."""
    env = _make_env(seed=0)
    # Zero out initial velocities for cleaner comparison
    env.unwrapped.lander.linearVelocity = (0.0, 0.0)
    env.unwrapped.lander.angularVelocity = 0.0

    s0 = _world_state(env)
    euler_state = dict(s0)

    Fm = solver.THRUST_MAX * 0.5  # half thrust
    n_steps = 50
    pos_errors = []
    vel_errors = []

    for _ in range(n_steps):
        _apply_thrust_impulse(env, Fm, 0.0)
        _step_box2d_no_action(env)
        euler_state = _euler_step(euler_state, Fm, 0.0, DT)

        box2d = _world_state(env)
        pos_err = math.hypot(box2d["x"] - euler_state["x"],
                             box2d["y"] - euler_state["y"])
        vel_err = math.hypot(box2d["vx"] - euler_state["vx"],
                             box2d["vy"] - euler_state["vy"])
        pos_errors.append(pos_err)
        vel_errors.append(vel_err)

    max_pos = max(pos_errors)
    max_vel = max(vel_errors)
    print(f"\nMain thrust: max_pos_err={max_pos:.6f}  max_vel_err={max_vel:.6f}")
    assert max_pos < 1.0, f"Main thrust pos divergence: {max_pos}"
    env.close()


def test_side_thrust_divergence():
    """Constant side thrust: measures side-engine force direction mismatch.

    The solver models Fs in direction (cosθ, sinθ), but Box2D applies it
    in direction (cosθ, -sinθ).  At θ≈0 these agree; as θ grows the
    y-component divergence appears.
    """
    env = _make_env(seed=0)
    env.unwrapped.lander.linearVelocity = (0.0, 0.0)
    env.unwrapped.lander.angularVelocity = 0.0
    # Tilt the lander ~15° to expose the sinθ sign mismatch
    env.unwrapped.lander.angle = 0.26

    s0 = _world_state(env)
    euler_state = dict(s0)

    Fs = solver.SIDE_MAX * 0.5
    n_steps = 50
    pos_errors = []
    angle_errors = []

    for _ in range(n_steps):
        _apply_thrust_impulse(env, 0.0, Fs)
        _step_box2d_no_action(env)
        euler_state = _euler_step(euler_state, 0.0, Fs, DT)

        box2d = _world_state(env)
        pos_err = math.hypot(box2d["x"] - euler_state["x"],
                             box2d["y"] - euler_state["y"])
        angle_err = abs(box2d["theta"] - euler_state["theta"])
        pos_errors.append(pos_err)
        angle_errors.append(angle_err)

    max_pos = max(pos_errors)
    max_angle = max(angle_errors)
    print(f"\nSide thrust (θ₀=0.26): max_pos_err={max_pos:.6f}  "
          f"max_angle_err={max_angle:.6f} rad")
    # This WILL be larger due to the sinθ sign mismatch
    env.close()


def test_solver_plan_divergence():
    """Solve a trajectory, replay thrusts in Box2D, measure drift.

    This is the end-to-end test: does the solver's plan actually work
    when executed in the real simulation?
    """
    env = _make_env(seed=42)
    # Zero out initial velocity to match solver assumption
    env.unwrapped.lander.linearVelocity = (0.0, 0.0)
    env.unwrapped.lander.angularVelocity = 0.0

    b2d = _world_state(env)
    start = np.array([b2d["x"], b2d["y"], b2d["theta"]])
    goal = np.array([solver.PAD_X, solver.PAD_Y, 0.0])

    # Solve
    times, plan, *_ = solver.solve(
        start=start, goal=goal,
        obstacles=(), time_budget=5.0, warmstart_budget=1.0,
    )
    duration = times[-1] - times[0]
    n_sim_steps = int(duration / DT)

    # Resample plan at DT intervals
    sim_times = np.linspace(times[0], times[-1], n_sim_steps)
    Fm_interp = np.interp(sim_times, times, plan["Fm"])
    Fs_interp = np.interp(sim_times, times, plan["Fs"])
    x_plan = np.interp(sim_times, times, plan["x"])
    y_plan = np.interp(sim_times, times, plan["y"])
    theta_plan = np.interp(sim_times, times, plan["theta"])

    pos_errors = []
    angle_errors = []

    for i in range(n_sim_steps):
        Fm_i = float(np.clip(Fm_interp[i], 0, solver.THRUST_MAX))
        Fs_i = float(np.clip(Fs_interp[i], -solver.SIDE_MAX, solver.SIDE_MAX))

        _apply_thrust_impulse(env, Fm_i, Fs_i)
        _step_box2d_no_action(env)

        box2d = _world_state(env)
        pos_err = math.hypot(box2d["x"] - x_plan[i],
                             box2d["y"] - y_plan[i])
        angle_err = abs(box2d["theta"] - theta_plan[i])
        pos_errors.append(pos_err)
        angle_errors.append(angle_err)

    final_box2d = _world_state(env)
    goal_err = math.hypot(final_box2d["x"] - goal[0],
                          final_box2d["y"] - goal[1])

    print(f"\nSolver plan replay ({n_sim_steps} steps, {duration:.2f}s):")
    print(f"  Position drift:  max={max(pos_errors):.4f}  "
          f"mean={np.mean(pos_errors):.4f}  final={pos_errors[-1]:.4f}")
    print(f"  Angle drift:     max={max(angle_errors):.4f}  "
          f"mean={np.mean(angle_errors):.4f}  final={angle_errors[-1]:.4f}")
    print(f"  Final goal dist: {goal_err:.4f}")

    env.close()


def test_mass_inertia_match():
    """Verify the hardcoded mass/inertia constants match what Box2D computes."""
    env = _make_env(seed=0)
    lander = env.unwrapped.lander
    legs = env.unwrapped.legs

    b2d_lander_mass = lander.mass
    b2d_lander_inertia = lander.inertia
    b2d_total_mass = b2d_lander_mass + sum(leg.mass for leg in legs)
    b2d_cm = lander.localCenter

    print(f"\nMass comparison:")
    print(f"  Solver MASS={solver.MASS:.6f}  Box2D total={b2d_total_mass:.6f}  "
          f"delta={abs(solver.MASS - b2d_total_mass):.6f}")
    print(f"  Solver INERTIA={solver.INERTIA:.6f}  Box2D lander={b2d_lander_inertia:.6f}  "
          f"delta={abs(solver.INERTIA - b2d_lander_inertia):.6f}")
    print(f"  Solver CM_LOCAL={ll.LANDER_CM_LOCAL}  "
          f"Box2D localCenter=({b2d_cm.x:.6f}, {b2d_cm.y:.6f})")

    assert abs(solver.MASS - b2d_total_mass) < 0.01, "Mass mismatch"
    assert abs(solver.INERTIA - b2d_lander_inertia) < 0.01, "Inertia mismatch"
    assert abs(ll.LANDER_CM_LOCAL[1] - b2d_cm.y) < 0.01, "CM y mismatch"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
