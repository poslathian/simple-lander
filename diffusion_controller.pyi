"""Simplified diffusion controller — position splines + PD tracking.

Three modules:
  DiffusionController — Top-level wiring, the thing lunar_lander.py imports.
  DiffusionModel      — Neural network: 21-dim conditioning -> 30-dim position CPs.
  DiffusionAction     — PD tracking controller that enforces guidance margin.

Conditioning: 21 dims (20 state + 1 CFG).
Model output: 30 dims (10 CPs x 3 channels: x, y, theta).
"""

from enum import IntEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# ── Coordinate types ──────────────────────────────────────────────────────

Position = tuple[float, float, float]      # (x, y, theta)
Velocity = tuple[float, float, float]      # (x', y', theta')
ThrustVec = tuple[float, float]            # (v, h) each in [-1, 1]

# ── Outcome ───────────────────────────────────────────────────────────────

class Outcome(IntEnum):
    """Ternary episode outcome for classifier-free guidance."""
    FAIL = -1
    UNKNOWN = 0
    SUCCESS = 1

# ── Input types ───────────────────────────────────────────────────────────

class LanderState(NamedTuple):
    """Two consecutive position observations for C2 continuity."""
    t_sim: float
    q_now: Position       # current position at t0
    q_prev: Position      # position at t0 - dt (implies velocity)

class ObstacleRelative(NamedTuple):
    """Nearest collision geometry relative to current observation.
    Wired to (0, 0, 0) until obstacles are implemented.
    """
    dx: float             # obstacle.x - q_now.x
    dy: float             # obstacle.y - q_now.y
    r: float              # obstacle radius

class WaypointTarget(NamedTuple):
    """Single waypoint: position and velocity error relative to current obs."""
    dq: Position          # (q_target - q_now): where to go
    dq_prime: Velocity    # (q'_target - q'_now): desired velocity delta

class GuidanceAction(NamedTuple):
    """Caller's suggested position at t_obs_cmd_latency."""
    q: Position           # suggested (x, y, theta) at next control step

# ── Output types ──────────────────────────────────────────────────────────

class PositionRef(NamedTuple):
    """Position reference at a single timestep."""
    x: float
    y: float
    theta: float

# ── Constants ─────────────────────────────────────────────────────────────

DT: float             # 0.02 — simulation timestep (50 Hz)
STATE_DIM: int        # 20
CFG_DIM: int          # 1
COND_DIM: int         # 21
X_DIM: int            # 30 (10 CPs x 3 channels)
N_CPS: int            # 10
N_CHANNELS: int       # 3 (x, y, theta)

# ── DiffusionModel ────────────────────────────────────────────────────────

class DiffusionModel:
    """Neural network: 21-dim conditioning -> 30-dim position CPs.

    Wraps MLP denoiser + DDIM sampler + CFG.
    Output is 10 control points x 3 (x, y, theta) in lander-relative coords.
    First CP pinned to (0,0,0), second CP constrained by q_prev for C1 continuity.
    """

    def __init__(self, checkpoint_path: str | None = ...) -> None: ...

    def predict(
        self,
        cond: NDArray[np.float32],
        outcome: Outcome,
        guidance_scale: float = ...,
    ) -> NDArray[np.float64]:
        """Run DDIM sampling with CFG.

        Args:
            cond: (20,) state conditioning vector (no CFG dim).
            outcome: desired outcome for CFG guidance.
            guidance_scale: CFG weight.

        Returns:
            (10, 3) position control points in lander-relative coords.
        """
        ...

# ── DiffusionAction ───────────────────────────────────────────────────────

class DiffusionAction:
    """PD tracking controller over a position spline.

    Evaluates the position spline at current time, computes position error,
    applies PD control to produce thrust commands. Enforces guidance margin
    by blending diffusion output with the caller's guidance suggestion.
    """

    def __init__(
        self,
        control_points: NDArray[np.float64],
        action_horizon: float,
        guidance: GuidanceAction,
        guidance_margin: float,
    ) -> None:
        """
        Args:
            control_points: (10, 3) from DiffusionModel.predict().
            action_horizon:  spline duration in seconds (default 1.5).
            guidance:        caller's suggested position at next step.
            guidance_margin: [0, 1] blend factor.
                             0 = pure diffusion, 1 = hard clamp to guidance.
        """
        ...

    def step(self, t: float, q_now: Position) -> ThrustVec:
        """Compute thrust command for current timestep.

        Args:
            t: time since spline start.
            q_now: current (x, y, theta) observation.

        Returns:
            (thrust_v, thrust_h) each in [-1, 1].
        """
        ...

    def position_ref(self, t: float) -> PositionRef:
        """Evaluate the position spline at time t (before PD)."""
        ...

# ── DiffusionController (top-level wiring) ────────────────────────────────

def DiffusionController(
    timeout: float,
    t_obs_cmd_latency: float,
    lander_state: LanderState,
    obstacle: ObstacleRelative,
    waypoint: WaypointTarget,
    guidance: GuidanceAction,
    guidance_margin: float,
    outcome: Outcome,
    action_horizon: float = ...,
) -> DiffusionAction:
    """Build conditioning, run DiffusionModel, return DiffusionAction.

    This is the single callable that lunar_lander.py imports.

    Args:
        timeout:             Episode timeout (seconds remaining).
        t_obs_cmd_latency:   Observation-to-command delay.
        lander_state:        q_now + q_prev (two observations for C2 continuity).
        obstacle:            Nearest collision geometry relative to obs (zeros for now).
        waypoint:            Target position/velocity error relative to obs.
        guidance:            Caller's suggested position at next step.
        guidance_margin:     [0, 1] how tightly to follow guidance.
        outcome:             Desired outcome for CFG (SUCCESS/FAIL/UNKNOWN).
        action_horizon:      Spline duration in seconds (default 1.5).

    Returns:
        DiffusionAction — call .step(t, q_now) each timestep for thrust.
    """
    ...
