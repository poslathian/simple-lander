"""DiffusionController — position-spline diffusion with PD tracking.

Three modules:
  DiffusionModel  — Neural network (or NoiseModel stub) -> position CPs.
  DiffusionAction — PD tracking controller + guidance margin enforcement.
  DiffusionController — Top-level wiring function.

Conditioning: 21 dims (20 state + 1 CFG).
Model output: 30 dims (10 CPs x 3 channels: x, y, theta).
"""

import math
from enum import IntEnum
from typing import NamedTuple, Protocol

import numpy as np
from scipy.interpolate import BSpline

import solver

# ── Constants ─────────────────────────────────────────────────────────────

DT = 0.02  # simulation timestep (50 Hz)
STATE_DIM = 20
CFG_DIM = 1
COND_DIM = STATE_DIM + CFG_DIM  # 21
N_CPS = 10
N_CHANNELS = 3  # x, y, theta
X_DIM = N_CPS * N_CHANNELS  # 30
DEGREE = 3  # cubic B-spline

# ── Coordinate types ──────────────────────────────────────────────────────

Position = tuple[float, float, float]      # (x, y, theta)
Velocity = tuple[float, float, float]      # (x', y', theta')
ThrustVec = tuple[float, float]            # (v, h) each in [-1, 1]

# ── Outcome ───────────────────────────────────────────────────────────────

class Outcome(IntEnum):
    FAIL = -1
    UNKNOWN = 0
    SUCCESS = 1

# ── Input types ───────────────────────────────────────────────────────────

class LanderState(NamedTuple):
    t_sim: float
    q_now: Position
    q_prev: Position

class ObstacleRelative(NamedTuple):
    dx: float
    dy: float
    r: float

class WaypointTarget(NamedTuple):
    dq: Position
    dq_prime: Velocity

class GuidanceAction(NamedTuple):
    q: Position           # suggested (x, y, theta) at next control step
    q_prime: Velocity     # suggested velocity (vx, vy, omega)
    q_double_prime: tuple[float, float, float]  # feedforward acceleration (ax, ay, alpha)

# ── Output types ──────────────────────────────────────────────────────────

class PositionRef(NamedTuple):
    x: float
    y: float
    theta: float


# ── Model protocol ────────────────────────────────────────────────────────

class ModelProtocol(Protocol):
    def predict(
        self,
        cond: np.ndarray,
        outcome: Outcome,
        guidance_scale: float,
    ) -> np.ndarray: ...


# ── NoiseModel (Step 0 stub) ─────────────────────────────────────────────

class NoiseModel:
    """Drop-in for DiffusionModel that returns random CPs."""

    def predict(
        self,
        cond: np.ndarray,
        outcome: Outcome,
        guidance_scale: float = 2.0,
    ) -> np.ndarray:
        return np.random.randn(N_CPS, N_CHANNELS) * 0.1


# ── Position B-spline helper ─────────────────────────────────────────────

def _make_position_spline(
    cps: np.ndarray,
    action_horizon: float,
) -> tuple[BSpline, BSpline, BSpline]:
    """Build 3 clamped cubic B-splines (x, y, theta) from (N_CPS, 3) CPs.

    Returns a tuple of (spline_x, spline_y, spline_theta).
    """
    n = cps.shape[0]
    # Clamped knot vector: first/last (degree+1) repeated
    n_internal = n - DEGREE + 1
    internal = np.linspace(0, action_horizon, n_internal)
    knots = np.concatenate([
        np.full(DEGREE, 0.0),
        internal,
        np.full(DEGREE, action_horizon),
    ])
    return (
        BSpline(knots, cps[:, 0], DEGREE, extrapolate=False),
        BSpline(knots, cps[:, 1], DEGREE, extrapolate=False),
        BSpline(knots, cps[:, 2], DEGREE, extrapolate=False),
    )


# ── Conditioning vector builder ───────────────────────────────────────────

def _build_cond(
    t_obs_cmd_latency: float,
    lander_state: LanderState,
    obstacle: ObstacleRelative,
    waypoint: WaypointTarget,
    guidance: GuidanceAction,
    action_horizon: float,
) -> np.ndarray:
    """Build the 20-dim state conditioning vector."""
    cond = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    # t_obs_cmd_latency (1)
    cond[idx] = t_obs_cmd_latency
    idx += 1

    # q_now (3)
    cond[idx:idx + 3] = lander_state.q_now
    idx += 3

    # q_prev (3)
    cond[idx:idx + 3] = lander_state.q_prev
    idx += 3

    # obstacle (3) — zeros for now
    cond[idx] = obstacle.dx
    cond[idx + 1] = obstacle.dy
    cond[idx + 2] = obstacle.r
    idx += 3

    # waypoint: delta_q (3) + delta_q' (3)
    cond[idx:idx + 3] = waypoint.dq
    idx += 3
    cond[idx:idx + 3] = waypoint.dq_prime
    idx += 3

    # guidance_q (3)
    cond[idx:idx + 3] = guidance.q
    idx += 3

    # action_horizon (1)
    cond[idx] = action_horizon
    idx += 1

    assert idx == STATE_DIM, f"Expected {STATE_DIM}, got {idx}"
    return cond


# ── DiffusionAction ───────────────────────────────────────────────────────

class DiffusionAction:
    """PD tracking controller over a position spline + guidance margin."""

    def __init__(
        self,
        control_points: np.ndarray,
        action_horizon: float,
        guidance: GuidanceAction,
        guidance_margin: float,
        q_now_world: Position,
        q_prev_world: Position,
    ):
        """
        Args:
            control_points: (10, 3) position CPs in lander-relative coords.
            action_horizon: spline duration in seconds.
            guidance: caller's suggested position at next step.
            guidance_margin: [0, 1]. 0=track guidance exactly, 1=trust model.
            q_now_world: current world-frame position (for converting refs).
            q_prev_world: previous world-frame position (for velocity estimate).
        """
        self.action_horizon = action_horizon
        self.guidance = guidance
        self.guidance_margin = np.clip(guidance_margin, 0.0, 1.0)
        self.q_now_world = np.array(q_now_world)
        self.q_prev_world = np.array(q_prev_world)
        self.gains = solver.DEFAULT_GAINS

        self._spline_x, self._spline_y, self._spline_theta = \
            _make_position_spline(control_points, action_horizon)

    def position_ref(self, t: float) -> PositionRef:
        """Evaluate position spline at time t (lander-relative)."""
        t_clamp = float(np.clip(t, 0.0, self.action_horizon))
        return PositionRef(
            x=float(self._spline_x(t_clamp)),
            y=float(self._spline_y(t_clamp)),
            theta=float(self._spline_theta(t_clamp)),
        )

    def step(
        self,
        t: float,
        q_now: Position,
        v_now: Velocity,
    ) -> ThrustVec:
        """Compute thrust via PD tracking of the blended position reference.

        Args:
            t: time since spline start.
            q_now: current (x, y, theta) in world frame.
            v_now: current (vx, vy, omega) in world frame.

        Returns:
            (thrust_v, thrust_h) each in [-1, 1].
        """
        # Model reference: spline is lander-relative, convert to world
        model_ref = self.position_ref(t)
        model_world = (
            self.q_now_world[0] + model_ref.x,
            self.q_now_world[1] + model_ref.y,
            self.q_now_world[2] + model_ref.theta,
        )

        # Guidance reference: already in world frame
        guide_world = self.guidance.q

        # Blend: margin=0 → guidance, margin=1 → model
        m = self.guidance_margin
        x_ref = (1 - m) * guide_world[0] + m * model_world[0]
        y_ref = (1 - m) * guide_world[1] + m * model_world[1]
        th_ref = (1 - m) * guide_world[2] + m * model_world[2]

        # Current state
        x, y, theta = q_now
        vx, vy, omega = v_now

        # Reference velocity and feedforward from guidance
        vx_ref = self.guidance.q_prime[0]
        vy_ref = self.guidance.q_prime[1]
        om_ref = self.guidance.q_prime[2]

        ax_ref = self.guidance.q_double_prime[0]
        ay_ref = self.guidance.q_double_prime[1]
        al_ref = self.guidance.q_double_prime[2]

        Fm, Fs = solver._tracking_step(
            x, y, theta, vx, vy, omega,
            x_ref, y_ref, th_ref, vx_ref, vy_ref, om_ref,
            ax_ref, ay_ref, al_ref, self.gains,
        )

        a_main = float(np.clip(2.0 * Fm / solver.THRUST_MAX - 1.0, -1.0, 1.0))
        a_side = float(np.clip(Fs / solver.SIDE_FORCE_MAX, -1.0, 1.0))

        return (a_main, a_side)


# ── DiffusionController (top-level wiring) ────────────────────────────────

_model_cache: dict = {}


def DiffusionController(
    timeout: float,
    t_obs_cmd_latency: float,
    lander_state: LanderState,
    obstacle: ObstacleRelative,
    waypoint: WaypointTarget,
    guidance: GuidanceAction,
    guidance_margin: float,
    outcome: Outcome,
    action_horizon: float = 1.5,
    model: ModelProtocol | None = None,
) -> DiffusionAction:
    """Build conditioning, run model, return DiffusionAction."""
    if model is None:
        if "model" not in _model_cache:
            _model_cache["model"] = NoiseModel()
        model = _model_cache["model"]

    cond = _build_cond(
        t_obs_cmd_latency=t_obs_cmd_latency,
        lander_state=lander_state,
        obstacle=obstacle,
        waypoint=waypoint,
        guidance=guidance,
        action_horizon=action_horizon,
    )

    cps = model.predict(cond, outcome, guidance_scale=2.0)

    # Pin first CP to origin (C0 continuity: we are at our own position)
    cps[0] = [0.0, 0.0, 0.0]

    return DiffusionAction(
        control_points=cps,
        action_horizon=action_horizon,
        guidance=guidance,
        guidance_margin=guidance_margin,
        q_now_world=lander_state.q_now,
        q_prev_world=lander_state.q_prev,
    )
