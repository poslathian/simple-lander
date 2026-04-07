"""Simplified diffusion controller interface — position splines + PD tracking.

Single waypoint, single guidance action, single ternary outcome.
Output is a position reference spline (x, y, theta) in lander-relative coords.
Caller uses PD tracking to convert position refs to thrust commands.

Conditioning vector: 32 dims (31 state + 1 CFG).
Model output: 30 dims (10 CPs x 3 channels).
"""

from enum import IntEnum
from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray

# ── Coordinate types ──────────────────────────────────────────────────────

Position = tuple[float, float, float]      # (x, y, theta)
Velocity = tuple[float, float, float]      # (x', y', theta')
ThrustVec = tuple[float, float]            # (v, h) each in [-1, 1]
Contacts = tuple[bool, bool, bool]         # (left_leg, right_leg, body)

# ── Outcome (replaces CFGItem union) ─────────────────────────────────────

class Outcome(IntEnum):
    """Ternary episode outcome for classifier-free guidance."""
    FAIL = -1
    UNKNOWN = 0
    SUCCESS = 1

# ── Input types ──────────────────────────────────────────────────────────

class LanderState(NamedTuple):
    """Current lander observation."""
    t_sim: float
    q: Position
    q_prime: Velocity
    thrust: ThrustVec
    contacts: Contacts

class WaypointTarget(NamedTuple):
    """Single position waypoint the trajectory should pass through."""
    t: float                  # time to reach waypoint (relative to now)
    t_margin: float           # acceptable timing slack
    q: Position               # target position (lander-relative)
    q_margin: Position        # acceptable position error per axis
    q_prime: Velocity         # target velocity at waypoint
    q_prime_margin: Velocity  # acceptable velocity error per axis

class GuidanceAction(NamedTuple):
    """Single thrust guidance hint (e.g., from KTO controller)."""
    thrust_v: float           # suggested vertical thrust
    thrust_v_margin: float    # how tightly to follow
    thrust_h: float           # suggested horizontal thrust
    thrust_h_margin: float    # how tightly to follow
    thrust_t: float           # when this guidance applies
    thrust_t_margin: float    # timing tolerance

# ── Output type ──────────────────────────────────────────────────────────

class PositionRef(NamedTuple):
    """Position reference at a single timestep."""
    x: float
    y: float
    theta: float

PositionSpline = Callable[[float], PositionRef]
"""Callable: time -> (x, y, theta) in lander-relative coordinates."""

# ── Constants ────────────────────────────────────────────────────────────

DT: float             # 0.02 — simulation timestep (50 Hz)
STATE_DIM: int        # 31
CFG_DIM: int          # 1
COND_DIM: int         # 32
X_DIM: int            # 30 (10 CPs x 3 channels)
N_CPS: int            # 10
N_CHANNELS: int       # 3 (x, y, theta)

# ── Main entry point ─────────────────────────────────────────────────────

def DiffusionController(
    timeout: float,
    t_obs_cmd_latency: float,
    lander_state: LanderState,
    waypoint: WaypointTarget,
    guidance: GuidanceAction,
    outcome: Outcome,
    action_horizon: float,
) -> PositionSpline:
    """Run diffusion inference and return a position reference spline.

    Builds a 32-dim conditioning vector, runs DDIM sampling with CFG
    (guided by outcome), and converts the 30-dim output into a callable
    position spline in lander-relative coordinates.

    The caller is responsible for PD tracking to convert the position
    reference into thrust commands.

    Args:
        timeout:             Episode timeout (seconds remaining).
        t_obs_cmd_latency:   Observation-to-command latency.
        lander_state:        Current lander state observation.
        waypoint:            Single target waypoint.
        guidance:            Single thrust guidance hint.
        outcome:             Desired outcome for CFG (SUCCESS/FAIL/UNKNOWN).
        action_horizon:      Duration of the output spline (seconds).

    Returns:
        A callable PositionSpline: f(t) -> PositionRef(x, y, theta).
    """
    ...
