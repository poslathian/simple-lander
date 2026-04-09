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

# ── Normalization: physics-based, max_velocity × action_horizon ─────────
# Each channel's scale = max achievable displacement over one action horizon.
# KTO velocity bounds: vx ∈ [-20,20], vy ∈ [-20,1], omega ∈ [-3,3].
# Action horizon = 1.5s. So max displacement = max_abs_vel × 1.5.
# x,y CPs are lander-relative displacements. theta CPs are world radians.
ACTION_HORIZON = 1.5
VEL_MAX_X = 20.0   # from solver velocity bounds
VEL_MAX_Y = 20.0
OMEGA_MAX = 3.0
NORM_SCALES = np.array([
    VEL_MAX_X * ACTION_HORIZON,   # 30.0 — max x displacement
    VEL_MAX_Y * ACTION_HORIZON,   # 30.0 — max y displacement
    OMEGA_MAX * ACTION_HORIZON,   #  4.5 — max theta change
], dtype=np.float64)

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
    """Drop-in for untrained DiffusionModel — unit Gaussian CPs.

    Matches the distribution of an untrained diffusion model's output:
    DDIM starts from randn, untrained model doesn't denoise, so output ≈ N(0,1).
    """

    def predict(
        self,
        cond: np.ndarray,
        outcome: Outcome,
        guidance_scale: float = 2.0,
    ) -> np.ndarray:
        return np.random.randn(N_CPS, N_CHANNELS)


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

        # Temporal consistency: stash previous raw prediction for comparison
        self._prev_splines: tuple | None = None
        self._prev_t0: float = 0.0
        self._prev_q_origin: np.ndarray = np.zeros(3)
        self._consistency_scores: list[float] = []  # per-inference consistency

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

        cps_norm = self.model.predict(cond, self.outcome, guidance_scale=2.0)
        cps_norm = np.clip(cps_norm, -1.0, 1.0)  # keep CPs in valid normalized range
        cps_norm[0, :2] = 0.0                    # x,y: pin to relative origin
        cps_norm[0, 2] = q_now[2] / NORM_SCALES[2]  # theta: pin to current world angle
        self._last_cps_norm = cps_norm.copy()

        # Stash previous prediction before overwriting
        self._prev_splines = self._diff_splines
        self._prev_t0 = self._diff_t0
        self._prev_q_origin = self._diff_q_origin.copy()

        # Denormalize to world units for spline building
        # x,y: relative world units. theta: world radians.
        cps_world = cps_norm * NORM_SCALES
        self._diff_splines = _make_position_spline(cps_world, self.action_horizon)
        self._diff_t0 = t_sim
        self._diff_q_origin = np.array(q_now)
        self._last_inference_q = q_now

        # Compute temporal consistency with previous prediction
        score = self._temporal_consistency()
        if score is not None:
            self._consistency_scores.append(score)

    def _temporal_consistency(self, compare_duration: float = 0.666) -> float | None:
        """Compare current prediction with previous on overlapping time window.

        Both predictions produce world-frame trajectories. We compare them
        on the overlap: the previous prediction evaluated at
        [dt, dt+compare_duration] vs the new prediction at [0, compare_duration],
        where dt = time between inference calls.

        Returns mean position error in world units, or None if no comparison.
        """
        if self._prev_splines is None or self._diff_splines is None:
            return None

        dt = self._diff_t0 - self._prev_t0  # time between inference calls
        if dt < DT or dt > self.action_horizon:
            return None

        n_samples = max(5, int(compare_duration / DT))
        errors = []

        for i in range(n_samples):
            t_rel = i * DT
            if t_rel > compare_duration:
                break

            # Previous prediction: evaluate at (dt + t_rel) from its t0
            t_prev = dt + t_rel
            if t_prev > self.action_horizon:
                break
            prev_rel = _eval_spline(self._prev_splines, t_prev, self.action_horizon)
            prev_world = (
                self._prev_q_origin[0] + prev_rel[0],
                self._prev_q_origin[1] + prev_rel[1],
                prev_rel[2],  # theta is world radians
            )

            # Current prediction: evaluate at t_rel from its t0
            curr_rel = _eval_spline(self._diff_splines, t_rel, self.action_horizon)
            curr_world = (
                self._diff_q_origin[0] + curr_rel[0],
                self._diff_q_origin[1] + curr_rel[1],
                curr_rel[2],
            )

            # Position error in world units
            dx = prev_world[0] - curr_world[0]
            dy = prev_world[1] - curr_world[1]
            errors.append(math.sqrt(dx * dx + dy * dy))

        return float(np.mean(errors)) if errors else None

    def get_action(self, guidance_margin: float = 0.001) -> ThrustVec:
        """Clamp diffusion position ref to within margin of KTO, PD track.

        Args:
            guidance_margin: [0,1] fraction of full range per channel.
                0=pure KTO, 0.5=half screen deviation, 1=unclamped.

        Returns:
            (thrust_v, thrust_h) each in [-1, 1].
        """
        uw = self.env.unwrapped
        L = uw.lander
        t_sim = uw.elapsed_s

        # Current state from Box2D
        x, y, theta = L.position.x, L.position.y, L.angle
        vx, vy, omega = L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity

        # KTO reference — zeros when exhausted
        kto_ref = self._get_kto_ref(t_sim)
        if kto_ref is not None:
            kto_q = kto_ref["q"]
            kto_v = kto_ref["v"]
            kto_a = kto_ref["a"]
        else:
            kto_q = (x, y, theta)
            kto_v = (0.0, 0.0, 0.0)
            kto_a = (0.0, 0.0, 0.0)

        # Weighted average: ref = (1-m)*kto + m*diffusion
        # At m=0 pure KTO, m=1 pure diffusion. Smooth, proportional blending.
        m = float(np.clip(guidance_margin, 0.0, 1.0))
        if self._diff_splines is not None and m > 0.0:
            t_diff = t_sim - self._diff_t0
            diff_rel = _eval_spline(self._diff_splines, t_diff, self.action_horizon)
            diff_world = np.array([
                self._diff_q_origin[0] + diff_rel[0],   # x: relative → world
                self._diff_q_origin[1] + diff_rel[1],   # y: relative → world
                diff_rel[2],                              # theta: already world radians
            ])
            x_ref = (1 - m) * kto_q[0] + m * diff_world[0]
            y_ref = (1 - m) * kto_q[1] + m * diff_world[1]
            th_ref = (1 - m) * kto_q[2] + m * diff_world[2]
        else:
            x_ref, y_ref, th_ref = kto_q

        # Velocity and accel refs come from KTO (zeros when exhausted)
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
