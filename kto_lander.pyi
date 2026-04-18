"""Type stubs for kto_lander — KTO-based tracking controller for LunarLander-v3."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Physics constants (module-level)
# ---------------------------------------------------------------------------
FPS: int
DT: float
SCALE: float

VIEWPORT_W_PX: int
VIEWPORT_H_PX: int
WORLD_W: float
WORLD_H: float

GRAVITY: float

MAIN_ENGINE_POWER: float
SIDE_ENGINE_POWER: float
MAIN_ENGINE_Y_OFF: float
SIDE_ENGINE_AWAY: float
SIDE_ENGINE_HEIGHT: float

LEG_DOWN_M: float
HELIPAD_Y_NOMINAL: float
PAD_BODY_Y: float

MAIN_THRUST_MAX: float
SIDE_THRUST_MAX: float
SIDE_TORQUE_MAX: float

M_POWER_MIN: float
S_POWER_MIN: float
MAIN_THRUST_ON_MIN: float
SIDE_TORQUE_ON_MIN: float

# ---------------------------------------------------------------------------
# Observation / state conversion
# ---------------------------------------------------------------------------
def obs_to_state(obs: np.ndarray) -> np.ndarray:
    """Convert gym obs (8,) to SI state [x, y, vx, vy, theta, omega] (6,), pad-centered."""
    ...

# ---------------------------------------------------------------------------
# Dynamics parameters
# ---------------------------------------------------------------------------
@dataclass
class LanderParams:
    mass: float
    inertia: float
    F_main_max: float = ...
    F_side_max: float = ...
    tau_side_max: float = ...
    g: float = ...

def compound_inertia_about_body_com(env: gym.Env) -> tuple[float, float]:
    """Return (total_mass, total_inertia) of body + legs via parallel-axis theorem."""
    ...

# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------
def randomize_initial_pose(env: gym.Env) -> None:
    """Teleport lander to a random (x, y) after env.reset()."""
    ...

def warmup_and_snapshot(
    env: gym.Env, n_steps: int = 5, randomize: bool = True
) -> tuple[np.ndarray, np.ndarray, LanderParams]:
    """Reset env, optionally randomize, step n_steps of noop, return (obs, state, params)."""
    ...

# ---------------------------------------------------------------------------
# Trajectory plan
# ---------------------------------------------------------------------------
@dataclass
class Plan:
    """KTO plan wrapping a 2D BsplineTrajectory. Callable: t -> 9-vector
    [x, y, vx, vy, theta, omega, ax, ay, alpha] via differential flatness."""
    traj: object       # pydrake.trajectories.BsplineTrajectory
    T: float
    g: float = ...

    def __call__(self, t: float) -> np.ndarray:
        """Evaluate the plan at time t, returning shape-(9,) reference state."""
        ...

def plan_with_kto(
    x0: float,
    y0: float,
    vx0: float,
    vy0: float,
    params: LanderParams,
    vy_target: float = -0.3,
    num_control_points: int = 12,
    spline_order: int = 4,  # B-spline order (degree + 1); 4 = cubic
    T_min: float = 1.5,
    T_max: float = 10.0,
    verbose: bool = True,
) -> tuple[Plan, float, object]:
    """Solve a pydrake KTO for a 2D descent to the pad.

    Returns (plan, duration_T, solver_result).
    Raises RuntimeError on solver failure.
    """
    ...

# ---------------------------------------------------------------------------
# Rendering overlay
# ---------------------------------------------------------------------------
def draw_plan_overlay(
    env: gym.Env,
    plan: Plan,
    current_t: float | None = None,
    n_samples: int = 80,
) -> None:
    """Draw the KTO path, knot dots, and current target onto the pygame screen."""
    ...

# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
@dataclass
class TrackerGains:
    kp_x: float = ...
    kd_x: float = ...
    kp_y: float = ...
    kd_y: float = ...
    kp_theta: float = ...
    kd_theta: float = ...
    theta_des_max: float = ...

def control(
    state: np.ndarray,
    t: float,
    plan: Plan,
    params: LanderParams,
    gains: TrackerGains,
    ff_only: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute action [-1,1]^2 and debug dict from current state and plan reference."""
    ...

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(
    render: bool = False,
    duration_s: float = 3.0,
    seed: int | None = None,
    verbose: bool = True,
    ref_kind: str = "kto",
    ff_only: bool = False,
    brake_time: float = 2.0,
) -> dict[str, float | int | bool]:
    """Run one episode. Returns metrics dict with keys:
    steps, terminated, estop, landed, success, window_closed,
    final_x, final_y, max_abs_{x,y,th}_err, rms_{x,y,th}_err.
    """
    ...

def main() -> None: ...
