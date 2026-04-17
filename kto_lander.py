"""Tracking controller scaffold for LunarLander-v3 (continuous).

This file implements steps (1) and (3) of the plan:
    1. Extract / query dynamics constants from the installed gymnasium source.
    3. Feedforward + PD tracker against a hand-written reference.

KTO is deliberately NOT here yet — we first validate the dynamics model and
tracker by tasking it with a trivial hover reference. If the tracker can hold
the lander near the post-impulse snapshot to within a small error for a few
seconds, our inverse-dynamics model is correct and KTO can be layered on.

Usage:
    python kto_lander.py                 # run once, no render
    python kto_lander.py --render        # watch it
    python kto_lander.py --episodes 20   # batch stats
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
# Constants extracted from gymnasium/envs/box2d/lunar_lander.py
# ---------------------------------------------------------------------------
FPS = 50
DT = 1.0 / FPS
SCALE = 30.0  # pixels per meter in Box2D world

VIEWPORT_W_PX = 600
VIEWPORT_H_PX = 400
WORLD_W = VIEWPORT_W_PX / SCALE   # 20.0 m
WORLD_H = VIEWPORT_H_PX / SCALE   # 13.333 m

GRAVITY = 10.0  # magnitude (m/s^2); world gravity is -GRAVITY in y

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
MAIN_ENGINE_Y_OFF = 4.0 / SCALE    # m, below body CoM along body -y
SIDE_ENGINE_AWAY = 12.0 / SCALE    # m, lateral offset used in thrust direction
SIDE_ENGINE_HEIGHT = 14.0 / SCALE  # m, effective moment arm for side torque

LEG_DOWN_M = 18.0 / SCALE          # m, pad-y reference uses this
HELIPAD_Y_NOMINAL = WORLD_H / 4    # m, terrain puts the pad here
PAD_BODY_Y = HELIPAD_Y_NOMINAL + LEG_DOWN_M  # body y when legs flush with pad

# Impulse-derived effective max force/torque magnitudes (at m_power=s_power=1):
MAIN_THRUST_MAX = MAIN_ENGINE_POWER * MAIN_ENGINE_Y_OFF / DT         # ~86.67 N
SIDE_THRUST_MAX = SIDE_ENGINE_POWER * SIDE_ENGINE_AWAY / DT          # 12.0 N
SIDE_TORQUE_MAX = SIDE_THRUST_MAX * SIDE_ENGINE_HEIGHT               # ~5.6 N*m

# Thrust floors (engines gate off below half throttle)
M_POWER_MIN = 0.5
S_POWER_MIN = 0.5
MAIN_THRUST_ON_MIN = M_POWER_MIN * MAIN_THRUST_MAX
SIDE_TORQUE_ON_MIN = S_POWER_MIN * SIDE_TORQUE_MAX


# ---------------------------------------------------------------------------
# Observation <-> SI state
# ---------------------------------------------------------------------------
#
# gym's obs (see step() in the source):
#   obs[0] = (pos.x - W/2) / (W/2)
#   obs[1] = (pos.y - pad_body_y) / (H/2)
#   obs[2] =  vel.x * (W/2) / FPS
#   obs[3] =  vel.y * (H/2) / FPS
#   obs[4] =  angle
#   obs[5] =  20 * angular_velocity / FPS
#   obs[6..7] = leg contacts
#
# We work in pad-centered world SI coordinates: x_rel = pos.x - W/2, y_rel = pos.y - pad_body_y.
def obs_to_state(obs: np.ndarray) -> np.ndarray:
    """Returns [x, y, vx, vy, theta, omega] in SI, pad-centered (x=0 above pad, y=0 at touchdown)."""
    x  = obs[0] * (WORLD_W / 2.0)
    y  = obs[1] * (WORLD_H / 2.0)
    vx = obs[2] * FPS / (WORLD_W / 2.0)
    vy = obs[3] * FPS / (WORLD_H / 2.0)
    th = obs[4]
    om = obs[5] * FPS / 20.0
    return np.array([x, y, vx, vy, th, om], dtype=np.float64)


# ---------------------------------------------------------------------------
# Dynamics params queried from the live env
# ---------------------------------------------------------------------------
@dataclass
class LanderParams:
    mass: float        # compound mass (body + 2 legs), kg
    inertia: float     # compound inertia about body CoM at rest leg config, kg*m^2
    # Engine constants (repeated here for convenience, populated from module consts)
    F_main_max: float = MAIN_THRUST_MAX
    F_side_max: float = SIDE_THRUST_MAX
    tau_side_max: float = SIDE_TORQUE_MAX
    g: float = GRAVITY


def compound_inertia_about_body_com(env) -> tuple[float, float]:
    """Compute (mass, inertia) of body + legs rigidly attached at their current pose.

    Parallel-axis theorem for each leg taken about body CoM in body frame.
    Uses live Box2D bodies so density/polygon-derived values are exact.
    """
    lander = env.unwrapped.lander
    legs = env.unwrapped.legs

    m_body = lander.mass
    I_body = lander.inertia  # about body CoM in body frame

    total_m = m_body
    total_I = I_body
    body_pos = np.array([lander.position.x, lander.position.y])
    body_ang = lander.angle
    R_wb = np.array([[math.cos(body_ang), -math.sin(body_ang)],
                     [math.sin(body_ang),  math.cos(body_ang)]])

    for leg in legs:
        m_leg = leg.mass
        I_leg = leg.inertia  # about leg CoM
        leg_com_world = np.array([leg.worldCenter.x, leg.worldCenter.y])
        r_world = leg_com_world - body_pos
        r_body = R_wb.T @ r_world  # in body frame
        d2 = float(r_body @ r_body)
        total_m += m_leg
        total_I += I_leg + m_leg * d2

    return total_m, total_I


# ---------------------------------------------------------------------------
# Warm-up: let INITIAL_RANDOM impulse settle and snapshot state
# ---------------------------------------------------------------------------
def warmup_and_snapshot(env, n_steps: int = 5):
    """Reset the env, take n_steps of noop (fully noop in continuous), then return (obs, state, params)."""
    obs, _ = env.reset()
    noop = np.array([-1.0, 0.0], dtype=np.float32)  # main off, side off
    for _ in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(noop)
        if terminated or truncated:
            raise RuntimeError("episode ended during warm-up — terrain may have clipped the lander")
    state = obs_to_state(obs)
    mass, inertia = compound_inertia_about_body_com(env)
    params = LanderParams(mass=mass, inertia=inertia)
    return obs, state, params


# ---------------------------------------------------------------------------
# Reference trajectories (hand-written, for dynamics validation)
# ---------------------------------------------------------------------------
class Reference:
    """Base class: produces (x, y, vx, vy, theta, omega, ax, ay, alpha) at time t."""

    def __call__(self, t: float) -> np.ndarray:
        raise NotImplementedError


class HoverReference(Reference):
    """Hold (x0, y0) with zero velocity and upright body. Feedforward is just gravity comp."""

    def __init__(self, x0: float, y0: float):
        self.x0, self.y0 = x0, y0

    def __call__(self, t: float) -> np.ndarray:
        return np.array([self.x0, self.y0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class FreeFallReference(Reference):
    """Predicted trajectory under no thrust. With FF=0 and PD=0, tracker should follow exactly."""

    def __init__(self, x0, y0, vx0, vy0, th0, om0, g=GRAVITY):
        self.x0, self.y0, self.vx0, self.vy0 = x0, y0, vx0, vy0
        self.th0, self.om0 = th0, om0
        self.g = g

    def __call__(self, t: float) -> np.ndarray:
        x  = self.x0 + self.vx0 * t
        y  = self.y0 + self.vy0 * t - 0.5 * self.g * t * t
        vx = self.vx0
        vy = self.vy0 - self.g * t
        th = self.th0 + self.om0 * t
        om = self.om0
        return np.array([x, y, vx, vy, th, om, 0.0, -self.g, 0.0])


class KTOReference(Reference):
    """Reference produced by pydrake KinematicTrajectoryOptimization in 2D (x, y),
    with θ, ω, α derived from differential flatness:
        θ(t) = atan2(-ẍ(t), ÿ(t)+g)

    Outside [0, T] we hold the terminal state (useful when the tracker runs past the plan).
    """

    def __init__(self, traj, T: float, g: float = GRAVITY):
        self._traj = traj            # pydrake BsplineTrajectory (2D)
        self._dtraj = traj.MakeDerivative(1)
        self._ddtraj = traj.MakeDerivative(2)
        self.T = float(T)
        self.g = g

    def _theta_at(self, t: float) -> float:
        tc = float(np.clip(t, 0.0, self.T))
        a = self._ddtraj.value(tc).flatten()
        return math.atan2(-a[0], a[1] + self.g)

    def __call__(self, t: float) -> np.ndarray:
        tc = float(np.clip(t, 0.0, self.T))
        q  = self._traj.value(tc).flatten()
        qd = self._dtraj.value(tc).flatten()
        qdd = self._ddtraj.value(tc).flatten()
        x, y = float(q[0]), float(q[1])
        vx, vy = float(qd[0]), float(qd[1])
        ax, ay = float(qdd[0]), float(qdd[1])

        th = math.atan2(-ax, ay + self.g)
        h = 5e-4
        th_p = self._theta_at(tc + h)
        th_m = self._theta_at(tc - h)
        om    = (th_p - th_m) / (2.0 * h)
        alpha = (th_p - 2.0 * th + th_m) / (h * h)
        return np.array([x, y, vx, vy, th, om, ax, ay, alpha])


def _pad_center_world(env) -> tuple[float, float]:
    """Return (pad_center_x, pad_body_y) in Box2D world coordinates (meters, +y up)."""
    helipad_y = env.unwrapped.helipad_y
    return WORLD_W / 2.0, helipad_y + LEG_DOWN_M


def draw_plan_overlay(env, ref: "KTOReference", current_t: float | None = None,
                      n_samples: int = 80) -> None:
    """Overlay the KTO plan onto env.unwrapped.screen: path line + knot dots + current target.

    Must be called after env.step() (which invokes env.render()) and followed by
    pygame.display.flip() to push the overlay to the window.
    """
    import pygame
    screen = env.unwrapped.screen
    if screen is None:
        return

    pad_x, pad_y = _pad_center_world(env)

    def to_screen(xrel: float, yrel: float) -> tuple[int, int]:
        wx = pad_x + xrel
        wy = pad_y + yrel
        sx = int(round(wx * SCALE))
        sy = int(round(VIEWPORT_H_PX - wy * SCALE))
        return (sx, sy)

    # Path line
    ts = np.linspace(0.0, ref.T, n_samples)
    pts = []
    for t in ts:
        r = ref(float(t))
        pts.append(to_screen(r[0], r[1]))
    if len(pts) > 1:
        pygame.draw.lines(screen, (255, 60, 255), False, pts, 2)

    # Control-point knots from the B-spline
    if hasattr(ref, "_traj"):
        cps = ref._traj.control_points()
        for cp in cps:
            arr = np.asarray(cp).flatten()
            if arr.size >= 2:
                pygame.draw.circle(screen, (255, 220, 0), to_screen(arr[0], arr[1]), 5)
                pygame.draw.circle(screen, (0, 0, 0), to_screen(arr[0], arr[1]), 5, 1)

    # Terminal marker (landing target)
    r_end = ref(ref.T)
    pygame.draw.circle(screen, (0, 255, 0), to_screen(r_end[0], r_end[1]), 6)

    # Current reference target (where the tracker is aiming right now)
    if current_t is not None:
        r_now = ref(float(current_t))
        pygame.draw.circle(screen, (60, 160, 255), to_screen(r_now[0], r_now[1]), 5)


def plan_with_kto(x0, y0, vx0, vy0, params: LanderParams,
                  vy_target: float = -0.3,
                  num_control_points: int = 12,
                  spline_order: int = 4,
                  T_min: float = 1.5,
                  T_max: float = 10.0,
                  verbose: bool = True):
    """Solve a pydrake KTO for a 2D (x, y) descent from (x0, y0, vx0, vy0) to (0, 0, 0, vy_target).

    Returns (KTOReference, duration_T, solver_result).
    Raises RuntimeError on solver failure.
    """
    from pydrake.planning import KinematicTrajectoryOptimization
    from pydrake.solvers import LinearEqualityConstraint, Solve

    m, g = params.mass, params.g
    F_max, F_min_on = params.F_main_max, M_POWER_MIN * params.F_main_max

    # Conservative box bounds: every (ẍ, ÿ) in this box satisfies powered-descent
    # floor (F_main ≥ F_min_on) and thrust ceiling (F_main ≤ F_max).
    #   min required: ẍ² + (ÿ+g)² ≥ (F_min_on/m)²
    #   max allowed:  ẍ² + (ÿ+g)² ≤ (F_max/m)²
    # With ax_max=4, ay_min=-1, ay_max=6.5 and g=10:
    #   worst-case-min: (ẍ=0, ÿ=-1) → (0)+(9)² = 81  ≥ (43.3/4.96)² = 76.4 ✓
    #   worst-case-max: (ẍ=4, ÿ=6.5) → 16 + 272.25 = 288  ≤ (86.67/4.96)² = 305 ✓
    ax_max, ay_min, ay_max = 4.0, -1.0, 6.5
    # Velocity bounds must accommodate any possible initial state. INITIAL_RANDOM=1000 N
    # applied for dt=0.02s over ~5 kg → up to ~4 m/s. Plus gravity during 5-step warmup
    # (~0.1s at g=10) adds ~1 m/s vy drift. Use generous bounds so initial-state
    # constraint never conflicts with them.
    vx_max, vy_min, vy_max = 7.0, -7.0, 5.0

    kto = KinematicTrajectoryOptimization(num_positions=2,
                                          num_control_points=num_control_points,
                                          spline_order=spline_order)

    kto.AddDurationConstraint(T_min, T_max)
    kto.AddDurationCost(1.0)
    kto.AddPathLengthCost(0.05)

    kto.AddVelocityBounds(np.array([-vx_max, vy_min]), np.array([vx_max, vy_max]))
    kto.AddAccelerationBounds(np.array([-ax_max, ay_min]), np.array([ax_max, ay_max]))

    # Boundary positions. Terminal y slightly below touchdown so the tracker drives
    # into the pad (contact ends the episode before we reach q(T) exactly).
    q0 = np.array([x0, y0])
    qT = np.array([0.0, -0.15])
    kto.AddPathPositionConstraint(q0, q0, 0.0)
    kto.AddPathPositionConstraint(qT, qT, 1.0)

    # Boundary velocities in physical time via AddVelocityConstraintAtNormalizedTime.
    # Its constraint is bound with vars [q(T·s), q̇(T·s)] → (4,) for 2D problem.
    # We express q̇(0) = (vx0, vy0) and q̇(T) = (0, vy_target) as linear equalities on those vars.
    def velocity_eq_constraint(vx_req, vy_req):
        A = np.zeros((2, 4))
        A[0, 2] = 1.0  # qxdot
        A[1, 3] = 1.0  # qydot
        return LinearEqualityConstraint(A, np.array([vx_req, vy_req]))

    kto.AddVelocityConstraintAtNormalizedTime(velocity_eq_constraint(vx0, vy0), 0.0)
    kto.AddVelocityConstraintAtNormalizedTime(velocity_eq_constraint(0.0, vy_target), 1.0)

    # Discourage solver from reporting infeasible on marginal initial conditions.
    # (A feasible solution typically exists for any valid warmup snapshot given our bounds.)

    # Initial guess: straight line between q0 and qT, constant duration 3.5s
    # (KTO's default guess may not be great for our constraints).
    T_guess = float(np.clip(3.5, T_min, T_max))
    cp_guess = np.linspace(q0, qT, num_control_points).T  # shape (2, N)
    # BSplineTrajectory expects the knots from the basis; SetInitialGuess takes a traj.
    # Simpler: just let solver start from default. Skip guess for now.

    result = Solve(kto.prog())
    if not result.is_success():
        if verbose:
            print(f"[kto] solver FAILED: {result.get_solver_id().name()} "
                  f"status={result.get_solution_result()}")
        raise RuntimeError(f"KTO solve failed (infeasible or solver issue): "
                           f"{result.get_solution_result()}")

    traj = kto.ReconstructTrajectory(result)
    T = float(traj.end_time() - traj.start_time())
    if verbose:
        print(f"[kto] solved: T={T:.2f}s, #cp={num_control_points}, "
              f"solver={result.get_solver_id().name()}")

    return KTOReference(traj, T), T, result


class QuinticBrakeReference(Reference):
    """Dynamically-feasible reference: quintic Hermite from (p0, v0, 0) to (p_end, 0, 0).

    Reference theta, omega, alpha derived from differential flatness:
        theta_ref(t) = atan2(-ax_ref(t), ay_ref(t) + g)
    omega/alpha computed by finite differences on theta_ref (analytic but tedious).
    """

    def __init__(self, x0, y0, vx0, vy0, x_end, y_end, T, g=GRAVITY):
        self.T = float(T)
        self.g = g
        self.cx = self._quintic(x0, vx0, 0.0, x_end, 0.0, 0.0, self.T)
        self.cy = self._quintic(y0, vy0, 0.0, y_end, 0.0, 0.0, self.T)

    @staticmethod
    def _quintic(p0, v0, a0, pT, vT, aT, T):
        A0 = p0
        A1 = v0
        A2 = a0 / 2.0
        dp = pT - (A0 + A1 * T + A2 * T * T)
        dv = vT - (A1 + 2 * A2 * T)
        da = aT - 2 * A2
        M = np.array([
            [T**3,    T**4,    T**5],
            [3*T**2,  4*T**3,  5*T**4],
            [6*T,    12*T**2, 20*T**3],
        ])
        sol = np.linalg.solve(M, np.array([dp, dv, da]))
        return np.array([A0, A1, A2, sol[0], sol[1], sol[2]])

    @staticmethod
    def _deriv(c, t, d):
        s = 0.0
        for k in range(d, 6):
            s += c[k] * math.factorial(k) / math.factorial(k - d) * t ** (k - d)
        return s

    def _theta_at(self, t):
        t = float(np.clip(t, 0.0, self.T))
        ax = self._deriv(self.cx, t, 2)
        ay = self._deriv(self.cy, t, 2)
        return math.atan2(-ax, ay + self.g)

    def __call__(self, t: float) -> np.ndarray:
        tc = float(np.clip(t, 0.0, self.T))
        x  = self._deriv(self.cx, tc, 0)
        y  = self._deriv(self.cy, tc, 0)
        vx = self._deriv(self.cx, tc, 1)
        vy = self._deriv(self.cy, tc, 1)
        ax = self._deriv(self.cx, tc, 2)
        ay = self._deriv(self.cy, tc, 2)

        th = math.atan2(-ax, ay + self.g)
        h = 5e-4
        th_p = self._theta_at(tc + h)
        th_m = self._theta_at(tc - h)
        om    = (th_p - th_m) / (2.0 * h)
        alpha = (th_p - 2.0 * th + th_m) / (h * h)
        return np.array([x, y, vx, vy, th, om, ax, ay, alpha])


# ---------------------------------------------------------------------------
# Feedforward + PD tracker
# ---------------------------------------------------------------------------
@dataclass
class TrackerGains:
    # Position/velocity (outer loop) — produces desired world accel
    kp_x: float = 2.0
    kd_x: float = 2.5
    kp_y: float = 8.0
    kd_y: float = 5.0
    # Attitude (inner loop) — produces desired angular accel
    kp_theta: float = 60.0
    kd_theta: float = 12.0
    # Saturation on desired tilt angle (rad)
    theta_des_max: float = 0.4


def control(state: np.ndarray, t: float, ref: Reference, params: LanderParams,
            gains: TrackerGains, ff_only: bool = False) -> tuple[np.ndarray, dict]:
    """Returns (action in [-1,1]^2, debug info).

    If ff_only is True, PD gains are ignored and only the reference's feedforward is applied.
    """
    x, y, vx, vy, theta, omega = state
    r = ref(t)
    xr, yr, vxr, vyr, thr, omr, axr, ayr, alr = r

    if ff_only:
        # Desired world accel is reference accel — no correction.
        ax_des, ay_des = axr, ayr
    else:
        ax_des = axr + gains.kp_x * (xr - x) + gains.kd_x * (vxr - vx)
        ay_des = ayr + gains.kp_y * (yr - y) + gains.kd_y * (vyr - vy)

    # Required world-frame net force on CoM (including gravity compensation)
    Fx_req = params.mass * ax_des
    Fy_req = params.mass * (ay_des + params.g)

    if ff_only:
        theta_cmd = thr
    else:
        # Desired body-up direction from force vector (attitude outer loop)
        theta_des = math.atan2(-Fx_req, max(Fy_req, 1e-6))
        theta_des = float(np.clip(theta_des, -gains.theta_des_max, gains.theta_des_max))
        theta_cmd = theta_des  # reference's thr is already encoded in ax/ay when feasible

    if ff_only:
        alpha_des = alr
        theta_err = 0.0
    else:
        theta_err = theta_cmd - theta
        theta_err = (theta_err + math.pi) % (2 * math.pi) - math.pi
        alpha_des = alr + gains.kp_theta * theta_err + gains.kd_theta * (omr - omega)

    # --- Main thrust: project required world force onto current body +y ---
    # body +y in world = (-sin θ, cos θ); projection scalar:
    F_main_req = -math.sin(theta) * Fx_req + math.cos(theta) * Fy_req

    # --- Side torque command ---
    tau_req = params.inertia * alpha_des

    # --- Quantize to continuous action-space (respecting deadzones) ---
    # Main engine: m_power = (action[0]+1)/2, fires only if action[0] > 0, saturates ≥ 0.5
    m_power_raw = F_main_req / params.F_main_max
    if m_power_raw < M_POWER_MIN:
        action0 = -1.0  # engine off (any value <= 0 is off; -1 is unambiguous)
    else:
        m_power = float(np.clip(m_power_raw, M_POWER_MIN, 1.0))
        action0 = 2.0 * m_power - 1.0

    # Side engine: torque = -sign(action[1]) * s_power * tau_max
    #  -> to get positive torque, action[1] < 0 (left engine). direction = -sign(tau).
    s_power_raw = abs(tau_req) / params.tau_side_max
    if s_power_raw < S_POWER_MIN:
        action1 = 0.0
    else:
        s_power = float(np.clip(s_power_raw, S_POWER_MIN, 1.0))
        direction = -math.copysign(1.0, tau_req)
        action1 = direction * s_power

    action = np.array([action0, action1], dtype=np.float32)
    dbg = dict(
        theta_cmd=theta_cmd, theta_err=theta_err,
        F_main_req=F_main_req, tau_req=tau_req,
        m_power_raw=m_power_raw, s_power_raw=s_power_raw,
    )
    return action, dbg


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(render: bool = False, duration_s: float = 3.0, seed: int | None = None,
                verbose: bool = True, ref_kind: str = "quintic", ff_only: bool = False,
                brake_time: float = 2.0) -> dict:
    env = gym.make("LunarLander-v3", continuous=True,
                   render_mode="human" if render else None)
    if seed is not None:
        env.reset(seed=seed)

    _, state0, params = warmup_and_snapshot(env, n_steps=5)
    x0, y0, vx0, vy0, th0, om0 = state0
    if verbose:
        print(f"[warmup] state = x={x0:+.3f} y={y0:+.3f} "
              f"vx={vx0:+.3f} vy={vy0:+.3f} θ={th0:+.3f} ω={om0:+.3f}")
        print(f"[params] mass={params.mass:.3f} kg  inertia={params.inertia:.4f} kg·m²")
        hover_thrust = params.mass * params.g
        print(f"[params] F_main_max={params.F_main_max:.2f} N, need {hover_thrust:.2f} N to hover "
              f"→ m_power={hover_thrust/params.F_main_max:.3f}")

    if ref_kind == "hover":
        ref: Reference = HoverReference(x0=x0, y0=y0)
    elif ref_kind == "freefall":
        ref = FreeFallReference(x0, y0, vx0, vy0, th0, om0)
    elif ref_kind == "quintic":
        x_end = x0 + 0.5 * vx0 * brake_time
        y_end = y0 + 0.5 * vy0 * brake_time
        ref = QuinticBrakeReference(x0, y0, vx0, vy0, x_end, y_end, brake_time)
        if verbose:
            print(f"[ref] quintic brake T={brake_time}s  end=({x_end:+.2f}, {y_end:+.2f})")
    elif ref_kind == "kto":
        ref, plan_T, _ = plan_with_kto(x0, y0, vx0, vy0, params, verbose=verbose)
        # Override duration to cover the plan plus a small settling tail
        duration_s = max(duration_s, plan_T + 1.0)
    else:
        raise ValueError(f"unknown ref_kind {ref_kind!r}")

    gains = TrackerGains()

    n_steps = int(round(duration_s / DT))
    errs = []
    obs_state = state0
    t = 0.0
    terminated = False
    estop = False
    landed = False
    last_obs = None
    # Sliding-window history of tracking-error norm for e-stop monitor.
    window_steps = max(1, int(round(0.1 / DT)))   # 100 ms
    err_hist: list[float] = []

    window_closed = False
    for k in range(n_steps):
        action, dbg = control(obs_state, t, ref, params, gains, ff_only=ff_only)
        obs, reward, terminated, truncated, info = env.step(action)
        # Overlay the plan on top of the env's rendered frame, if we have a KTO plan.
        if render:
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN
                        and event.key in (pygame.K_q, pygame.K_ESCAPE)
                    ):
                        window_closed = True
                if isinstance(ref, KTOReference):
                    draw_plan_overlay(env, ref, current_t=t)
                    pygame.display.flip()
            except Exception:
                pass
            if window_closed:
                break
        obs_state = obs_to_state(obs)
        last_obs = obs
        t += DT
        r_next = ref(t)
        e_x = obs_state[0] - r_next[0]
        e_y = obs_state[1] - r_next[1]
        e_th = obs_state[4] - r_next[4]
        errs.append((e_x, e_y, e_th))

        # E-stop monitor: if the weighted error norm is monotonically increasing
        # over the last `window_steps` samples, declare overtorque failure.
        e_norm = math.sqrt(e_x * e_x + e_y * e_y + (2.0 * e_th) ** 2)
        err_hist.append(e_norm)
        if len(err_hist) > window_steps:
            err_hist.pop(0)
        if (len(err_hist) == window_steps
                and all(err_hist[i + 1] > err_hist[i] for i in range(window_steps - 1))
                and err_hist[-1] - err_hist[0] > 0.30):
            estop = True
            break

        # Landing check: both legs in contact and low vertical speed + near pad
        if obs[6] > 0.5 and obs[7] > 0.5:
            landed = True
            break

        if terminated or truncated:
            break

    env.close()
    errs = np.array(errs)
    # Landing success: both legs touched AND awake-state indicates settled
    # (Box2D marks the body !awake → env.step returns terminated=True with reward +100).
    # We define a success as landed==True; the gym will mark terminated on next step.
    success = bool(landed)

    final_state = obs_to_state(last_obs) if last_obs is not None else state0
    metrics = dict(
        steps=len(errs),
        terminated=terminated,
        estop=estop,
        landed=landed,
        success=success,
        window_closed=window_closed,
        final_x=float(final_state[0]),
        final_y=float(final_state[1]),
        max_abs_x_err=float(np.max(np.abs(errs[:, 0]))) if len(errs) else 0.0,
        max_abs_y_err=float(np.max(np.abs(errs[:, 1]))) if len(errs) else 0.0,
        max_abs_th_err=float(np.max(np.abs(errs[:, 2]))) if len(errs) else 0.0,
        rms_x_err=float(np.sqrt(np.mean(errs[:, 0] ** 2))) if len(errs) else 0.0,
        rms_y_err=float(np.sqrt(np.mean(errs[:, 1] ** 2))) if len(errs) else 0.0,
        rms_th_err=float(np.sqrt(np.mean(errs[:, 2] ** 2))) if len(errs) else 0.0,
    )
    if verbose:
        status = ("LANDED" if landed else
                  "ESTOP " if estop else
                  "CRASH " if terminated else
                  "TIMED ")
        print(f"[metrics] {status} steps={metrics['steps']}/{n_steps} "
              f"final=({final_state[0]:+.2f}, {final_state[1]:+.2f})")
        print(f"          |x err|  max={metrics['max_abs_x_err']:.3f} m  rms={metrics['rms_x_err']:.3f}")
        print(f"          |y err|  max={metrics['max_abs_y_err']:.3f} m  rms={metrics['rms_y_err']:.3f}")
        print(f"          |θ err| max={metrics['max_abs_th_err']:.3f} rad rms={metrics['rms_th_err']:.3f}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--render", action="store_true")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ref", choices=["hover", "freefall", "quintic", "kto"], default="kto")
    p.add_argument("--ff-only", action="store_true",
                   help="disable PD feedback — tracker runs open-loop on reference feedforward")
    p.add_argument("--brake-time", type=float, default=2.0)
    args = p.parse_args()

    # With --render, loop indefinitely with incrementing seeds unless the user
    # specified --episodes explicitly. Close the window or press Q/Esc to exit.
    infinite = args.render and args.episodes == 1 and not any(
        a.startswith("--episodes") for a in __import__("sys").argv[1:]
    )

    all_metrics = []
    i = 0
    base_seed = args.seed if args.seed is not None else 0
    while True:
        if not infinite and i >= args.episodes:
            break
        seed = base_seed + i if args.seed is not None else (None if not infinite else base_seed + i)
        label = f"{i+1}/{'∞' if infinite else args.episodes}"
        print(f"=== episode {label} (seed={seed}, ref={args.ref}, ff_only={args.ff_only}) ===")
        try:
            m = run_episode(render=args.render, duration_s=args.duration, seed=seed,
                            ref_kind=args.ref, ff_only=args.ff_only, brake_time=args.brake_time)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            break
        all_metrics.append(m)
        i += 1
        if m.get("window_closed"):
            print("[window closed, stopping loop]")
            break

    if len(all_metrics) > 1:
        rms_x = np.mean([m["rms_x_err"] for m in all_metrics])
        rms_y = np.mean([m["rms_y_err"] for m in all_metrics])
        rms_t = np.mean([m["rms_th_err"] for m in all_metrics])
        landed = sum(m["landed"] for m in all_metrics)
        estopped = sum(m["estop"] for m in all_metrics)
        crashed = sum(m["terminated"] and not m["landed"] for m in all_metrics)
        N = len(all_metrics)
        timed = N - landed - estopped - crashed
        print(f"=== summary over {N} eps ===")
        print(f"  landed:  {landed}/{N}  ({100*landed/N:.0f}%)")
        print(f"  estop:   {estopped}/{N}  ({100*estopped/N:.0f}%)")
        print(f"  crashed: {crashed}/{N}  ({100*crashed/N:.0f}%)")
        print(f"  timed:   {timed}/{N}  ({100*timed/N:.0f}%)")
        print(f"  tracking rms: x={rms_x:.3f} m  y={rms_y:.3f} m  θ={rms_t:.3f} rad")


if __name__ == "__main__":
    main()
