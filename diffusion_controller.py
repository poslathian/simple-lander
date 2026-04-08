"""KTODiffusionController — KTO warm-start + diffusion refinement with PD tracking.

Stateful controller that owns both the KTO guidance spline and the diffusion
model. One PD tracking loop over a blended position spline.

Conditioning: 21 dims (20 state + 1 CFG).
Model output: 30 dims (10 CPs x 3 channels: x, y, theta).
"""

import math
from enum import IntEnum
from typing import NamedTuple, Protocol

import numpy as np
from scipy.interpolate import BSpline

import solver
from lunar_lander import KTOController, TIMEOUT

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

class ObstacleRelative(NamedTuple):
    dx: float
    dy: float
    r: float

class WaypointTarget(NamedTuple):
    dq: Position
    dq_prime: Velocity

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
    duration: float,
) -> tuple[BSpline, BSpline, BSpline]:
    """Build 3 clamped cubic B-splines (x, y, theta) from (N, 3) CPs."""
    n = cps.shape[0]
    n_internal = n - DEGREE + 1
    internal = np.linspace(0, duration, n_internal)
    knots = np.concatenate([
        np.full(DEGREE, 0.0),
        internal,
        np.full(DEGREE, duration),
    ])
    return (
        BSpline(knots, cps[:, 0], DEGREE, extrapolate=False),
        BSpline(knots, cps[:, 1], DEGREE, extrapolate=False),
        BSpline(knots, cps[:, 2], DEGREE, extrapolate=False),
    )


def _eval_spline(splines, t, duration):
    """Evaluate (spline_x, spline_y, spline_theta) at clamped time t."""
    t_c = float(np.clip(t, 0.0, duration))
    return (float(splines[0](t_c)), float(splines[1](t_c)), float(splines[2](t_c)))


# ── Conditioning vector builder ───────────────────────────────────────────

def _build_cond(
    t_obs_cmd_latency: float,
    q_now: Position,
    q_prev: Position,
    obstacle: ObstacleRelative,
    waypoint: WaypointTarget,
    guidance_q: Position,
    action_horizon: float,
) -> np.ndarray:
    """Build the 20-dim state conditioning vector."""
    cond = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    cond[idx] = t_obs_cmd_latency; idx += 1
    cond[idx:idx + 3] = q_now; idx += 3
    cond[idx:idx + 3] = q_prev; idx += 3
    cond[idx] = obstacle.dx; cond[idx+1] = obstacle.dy; cond[idx+2] = obstacle.r; idx += 3
    cond[idx:idx + 3] = waypoint.dq; idx += 3
    cond[idx:idx + 3] = waypoint.dq_prime; idx += 3
    cond[idx:idx + 3] = guidance_q; idx += 3
    cond[idx] = action_horizon; idx += 1

    assert idx == STATE_DIM, f"Expected {STATE_DIM}, got {idx}"
    return cond


# ── KTODiffusionController ────────────────────────────────────────────────

class KTODiffusionController:
    """Stateful controller: KTO warm-start + diffusion refinement + PD tracking.

    Lifecycle:
        1. __init__: create with env, model, config
        2. warm_start(): solve KTO, store guidance spline
        3. Loop:
           a. inference() at target_frequency (3 Hz)
           b. get_action() every sim step (50 Hz)
    """

    def __init__(
        self,
        env,
        model: ModelProtocol | None = None,
        target_frequency: float = 3.0,
        action_horizon: float = 1.5,
        outcome: Outcome = Outcome.SUCCESS,
    ):
        self.env = env
        self.model = model if model is not None else NoiseModel()
        self.target_frequency = target_frequency
        self.action_horizon = action_horizon
        self.outcome = outcome
        self.gains = solver.DEFAULT_GAINS

        # KTO state
        self._kto: KTOController | None = None
        self._kto_t0: float = 0.0  # t_sim when warm_start was called
        self._kto_duration: float = 0.0

        # Diffusion state
        self._diff_splines: tuple | None = None
        self._diff_t0: float = 0.0  # t_sim when inference was called
        self._diff_q_origin: np.ndarray = np.zeros(3)  # world pos at inference time
        self._last_inference_q: Position | None = None  # q_prev for next inference

    def warm_start(
        self,
        waypoint: WaypointTarget | None = None,
        time_budget: float = 5.0,
    ) -> None:
        """Solve KTO to produce guidance spline."""
        self._kto = KTOController(self.env, time_budget=time_budget)
        self._kto_t0 = self.env.unwrapped.elapsed_s
        self._kto_duration = self._kto.n_steps * DT

    def _get_kto_ref(self, t_sim: float):
        """Get KTO reference (pos, vel, accel) at t_sim. Returns None if exhausted."""
        if self._kto is None:
            return None
        t_kto = t_sim - self._kto_t0
        idx = int(round(t_kto / DT))
        if idx < 0 or idx >= self._kto.n_steps:
            return None
        p = self._kto.plan
        return {
            "q": (float(p["x"][idx]), float(p["y"][idx]), float(p["theta"][idx])),
            "v": (float(p["vx"][idx]), float(p["vy"][idx]), float(p["omega"][idx])),
            "a": (float(p["ax"][idx]), float(p["ay"][idx]), float(p["alpha"][idx])),
        }

    def inference(self) -> None:
        """Run diffusion model at current observation."""
        uw = self.env.unwrapped
        L = uw.lander
        t_sim = uw.elapsed_s
        q_now = (L.position.x, L.position.y, L.angle)

        q_prev = self._last_inference_q if self._last_inference_q is not None else q_now

        vel = (L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity)
        pad_x, pad_y = solver.PAD_X, solver.PAD_Y
        dq = (pad_x - q_now[0], pad_y - q_now[1], 0.0 - q_now[2])
        dq_prime = (0.0 - vel[0], -0.5 - vel[1], 0.0 - vel[2])

        # KTO reference at current time for guidance_q conditioning
        kto_ref = self._get_kto_ref(t_sim)
        guidance_q = kto_ref["q"] if kto_ref else q_now

        cond = _build_cond(
            t_obs_cmd_latency=DT,
            q_now=q_now,
            q_prev=q_prev,
            obstacle=ObstacleRelative(0.0, 0.0, 0.0),
            waypoint=WaypointTarget(dq=dq, dq_prime=dq_prime),
            guidance_q=guidance_q,
            action_horizon=self.action_horizon,
        )

        cps = self.model.predict(cond, self.outcome, guidance_scale=2.0)
        cps[0] = [0.0, 0.0, 0.0]  # Pin first CP to origin

        self._diff_splines = _make_position_spline(cps, self.action_horizon)
        self._diff_t0 = t_sim
        self._diff_q_origin = np.array(q_now)
        self._last_inference_q = q_now

    def get_action(self, guidance_margin: float = 0.001) -> ThrustVec:
        """Blend KTO + diffusion splines, PD track the result.

        Args:
            guidance_margin: [0,1]. 0=track KTO, 1=trust model.

        Returns:
            (thrust_v, thrust_h) each in [-1, 1].
        """
        uw = self.env.unwrapped
        L = uw.lander
        t_sim = uw.elapsed_s

        # Check if KTO plan is exhausted
        kto_ref = self._get_kto_ref(t_sim)
        if kto_ref is None:
            return (0.0, 0.0)  # zero thrust, gravity settles

        # Current state from Box2D
        x, y, theta = L.position.x, L.position.y, L.angle
        vx, vy, omega = L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity

        # KTO reference (world frame)
        kto_q = kto_ref["q"]
        kto_v = kto_ref["v"]
        kto_a = kto_ref["a"]

        # Diffusion reference (if available), clamped to within margin of KTO
        m = np.clip(guidance_margin, 0.0, 1.0)
        if self._diff_splines is not None and m > 0.0:
            t_diff = t_sim - self._diff_t0
            diff_rel = _eval_spline(self._diff_splines, t_diff, self.action_horizon)
            diff_world = (
                self._diff_q_origin[0] + diff_rel[0],
                self._diff_q_origin[1] + diff_rel[1],
                self._diff_q_origin[2] + diff_rel[2],
            )
            # Clamp: diffusion ref clamped to within margin distance of KTO
            x_ref = float(np.clip(diff_world[0], kto_q[0] - m, kto_q[0] + m))
            y_ref = float(np.clip(diff_world[1], kto_q[1] - m, kto_q[1] + m))
            th_ref = float(np.clip(diff_world[2], kto_q[2] - m, kto_q[2] + m))
        else:
            x_ref, y_ref, th_ref = kto_q

        # Velocity and accel refs come from KTO (model doesn't produce these)
        vx_ref, vy_ref, om_ref = kto_v
        ax_ref, ay_ref, al_ref = kto_a

        Fm, Fs = solver._tracking_step(
            x, y, theta, vx, vy, omega,
            x_ref, y_ref, th_ref, vx_ref, vy_ref, om_ref,
            ax_ref, ay_ref, al_ref, self.gains,
        )

        a_main = float(np.clip(2.0 * Fm / solver.THRUST_MAX - 1.0, -1.0, 1.0))
        a_side = float(np.clip(Fs / solver.SIDE_FORCE_MAX, -1.0, 1.0))

        return (a_main, a_side)
