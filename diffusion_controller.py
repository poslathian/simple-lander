"""DiffusionController — Noise surrogate conforming to the .pyi interface."""

import math
from typing import Annotated, Callable, Literal, NamedTuple

import numpy as np


# ── Coordinate types (mirror .pyi) ────────────────────────────────────────

Position = tuple[float, float, float]      # (x, y, theta)
Velocity = tuple[float, float, float]      # (x', y', theta')
ThrustVec = tuple[float, float]            # (v, h) each in [-1, 1]
Contacts = tuple[bool, bool, bool]         # (left_leg, right_leg, body)


class MaxLen:
    def __init__(self, n: int) -> None:
        self.n = n


# ── Input types ───────────────────────────────────────────────────────────

class Obstacle(NamedTuple):
    dx: float
    dy: float
    r: float


class LanderState(NamedTuple):
    t_sim_lander: float
    q: Position
    q_prime: Velocity
    thrust: ThrustVec
    contacts: Contacts


class WaypointTarget(NamedTuple):
    t: float
    t_margin: float
    q: Position
    q_margin: Position
    q_prime: Velocity
    q_prime_margin: Velocity


class ActionTarget(NamedTuple):
    thrust_v: float
    thrust_v_margin: float
    thrust_h: float
    thrust_h_margin: float
    thrust_t: float
    thrust_t_margin: float


class WaypointResult(NamedTuple):
    outcome: Literal[-1, 0, 1]
    alpha: float


class ActionResult(NamedTuple):
    outcome: Literal[-1, 0, 1]
    alpha: float


class ContactGuidance(NamedTuple):
    t_contact: float
    alpha: float


class CrashGuidance(NamedTuple):
    t_crash: float
    alpha: float


CFGItem = WaypointResult | ActionResult | ContactGuidance | CrashGuidance

ThrustSpline = Callable[[float], ThrustVec]

# Own RNG to avoid polluting env's random state
_rng = np.random.default_rng()


# ── Main entry point ──────────────────────────────────────────────────────

def DiffusionController(
    timeout: float,
    t_obs_cmd_latency: float,
    obstacles: Annotated[list[Obstacle], MaxLen(5)],
    lander_state: LanderState,
    waypoint_goals: Annotated[list[WaypointTarget], MaxLen(5)],
    guidance_actions: Annotated[list[ActionTarget], MaxLen(5)],
    classifier_free_guidance: Annotated[list[CFGItem], MaxLen(10)],
    action_horizon: float,
    target_frequency: float,
) -> ThrustSpline:
    """Return a ThrustSpline over [0, action_horizon].

    Pure noise surrogate: generates gaussian noise clipped to [-1, 1],
    then clamps to any matching guidance_actions within their margins.
    """
    dt_step = 1.0 / max(target_frequency, 1.0)
    n_steps = max(1, int(math.ceil(action_horizon / dt_step)))
    dt_step = action_horizon / n_steps

    times: list[float] = []
    actions: list[ThrustVec] = []

    for i in range(n_steps):
        t_rel = i * dt_step

        # Generate pure noise
        noise_v = float(np.clip(_rng.standard_normal(), -1.0, 1.0))
        noise_h = float(np.clip(_rng.standard_normal(), -1.0, 1.0))

        # Clamp to guidance actions that match this time step
        tv, th = noise_v, noise_h
        for at in guidance_actions:
            if abs(t_rel - at.thrust_t) <= at.thrust_t_margin:
                tv = float(np.clip(tv, at.thrust_v - at.thrust_v_margin,
                                   at.thrust_v + at.thrust_v_margin))
                th = float(np.clip(th, at.thrust_h - at.thrust_h_margin,
                                   at.thrust_h + at.thrust_h_margin))

        times.append(t_rel)
        actions.append((tv, th))

    _times = times
    _actions = actions
    _horizon = action_horizon

    def spline(t: float) -> ThrustVec:
        t = max(0.0, min(t, _horizon))
        if len(_actions) == 1:
            return _actions[0]
        idx = int(t / (_horizon / len(_actions)))
        idx = min(idx, len(_actions) - 1)
        return _actions[idx]

    return spline
