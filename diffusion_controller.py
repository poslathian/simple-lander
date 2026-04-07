"""DiffusionController — Diffusion-model-based thrust controller for Lunar Lander.

Loads a DiffusionMLP checkpoint and builds the 131-dim conditioning vector
from the ICD interface inputs, runs DDIM sampling with CFG, then converts
the 20-dim B-spline control points into a callable ThrustSpline.
"""

import math
import os
from typing import Annotated, Callable, Literal, NamedTuple

import numpy as np
import torch

from model import (
    DiffusionMLP, CosineSchedule, DDIMSampler,
    W, H, PAD_CX, PAD_Y, OBS_RADIUS,
    OBS_X_NORM_OFFSET, OBS_X_NORM_SCALE,
    OBS_Y_NORM_OFFSET, OBS_Y_NORM_SCALE,
    STATE_DIM, CFG_DIM, COND_DIM, X_DIM, CFG_START,
)
from thrust_spline import ThrustSpline as _ThrustSplineImpl


# ── Coordinate types (mirror .pyi) ──────────────────────────────────────────

Position = tuple[float, float, float]      # (x, y, theta)
Velocity = tuple[float, float, float]      # (x', y', theta')
ThrustVec = tuple[float, float]            # (v, h) each in [-1, 1]
Contacts = tuple[bool, bool, bool]         # (left_leg, right_leg, body)


class MaxLen:
    def __init__(self, n: int) -> None:
        self.n = n


# ── Input types ─────────────────────────────────────────────────────────────

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

DT = 0.02  # simulation timestep (1/FPS)


# ── Normalization helpers (matching adapter.py) ─────────────────────────────

_COND_BOUNDS = {
    "dx": (-W, W),
    "dy": (-H, H),
    "theta": (-math.pi, math.pi),
    "vx": (-5.0, 5.0),
    "vy": (-5.0, 5.0),
    "omega": (-5.0, 5.0),
    "thrust_v": (-1.0, 1.0),
    "thrust_h": (-1.0, 1.0),
    "contact": (0.0, 1.0),
    "obs_dx": (-W, W),
    "obs_dy": (-H, H),
    "obs_r": (0.0, 2.0),
    "action_horizon": (0.0, 10.0),
    "target_freq": (0.0, 50.0),
    "wp_pos": (-W, W),
    "wp_vel": (-5.0, 5.0),
    "wp_t": (0.0, 10.0),
    "wp_margin": (0.0, W),
    "act_thrust": (-1.0, 1.0),
    "act_margin": (0.0, 2.0),
    "act_t": (0.0, 10.0),
    "act_t_margin": (0.0, 1.0),
}


def _norm_clip(val: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return float(np.clip(2.0 * (val - lo) / (hi - lo) - 1.0, -1.0, 1.0))


# ── Lazy-loaded model singleton ─────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def _get_model():
    """Load model from DIFFUSION_MODEL_PATH checkpoint, or random weights as fallback."""
    if "sampler" in _MODEL_CACHE:
        return _MODEL_CACHE["sampler"], _MODEL_CACHE["norm_stats"]

    model = DiffusionMLP()

    ckpt_path = os.environ.get("DIFFUSION_MODEL_PATH", "")
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        norm_stats = {
            "x_mean": np.asarray(ckpt["x_mean"], dtype=np.float32),
            "x_std":  np.asarray(ckpt["x_std"],  dtype=np.float32),
        }
    else:
        # Fallback: identity normalization with random weights (for testing only)
        norm_stats = {
            "x_mean": np.zeros(X_DIM, dtype=np.float32),
            "x_std":  np.ones(X_DIM,  dtype=np.float32),
        }

    model.eval()
    schedule = CosineSchedule(T=100)
    sampler = DDIMSampler(model, schedule, n_steps=10)

    _MODEL_CACHE["sampler"] = sampler
    _MODEL_CACHE["norm_stats"] = norm_stats
    return sampler, norm_stats


# ── Conditioning vector builder ─────────────────────────────────────────────

def _build_cond(
    t_obs_cmd_latency: float,
    lander_state: LanderState,
    obstacles: list[Obstacle],
    waypoint_goals: list[WaypointTarget],
    guidance_actions: list[ActionTarget],
    classifier_free_guidance: list[CFGItem],
    action_horizon: float,
    target_frequency: float,
) -> np.ndarray:
    """Build the 131-dim conditioning vector matching adapter.py layout."""
    cond = np.zeros(COND_DIM, dtype=np.float32)
    idx = 0

    # t_obs_cmd_latency (1)
    cond[idx] = _norm_clip(t_obs_cmd_latency, 0.0, 0.5)
    idx += 1

    # Q (x, y, theta) — lander-relative (always 0,0,0 at observation time) (3)
    cond[idx] = _norm_clip(lander_state.q[0], *_COND_BOUNDS["dx"])
    cond[idx + 1] = _norm_clip(lander_state.q[1], *_COND_BOUNDS["dy"])
    cond[idx + 2] = _norm_clip(lander_state.q[2], *_COND_BOUNDS["theta"])
    idx += 3

    # Q' (vx, vy, omega) (3)
    cond[idx] = _norm_clip(lander_state.q_prime[0], *_COND_BOUNDS["vx"])
    cond[idx + 1] = _norm_clip(lander_state.q_prime[1], *_COND_BOUNDS["vy"])
    cond[idx + 2] = _norm_clip(lander_state.q_prime[2], *_COND_BOUNDS["omega"])
    idx += 3

    # Thrust (v, h) (2)
    cond[idx] = _norm_clip(lander_state.thrust[0], -1.0, 1.0)
    cond[idx + 1] = _norm_clip(lander_state.thrust[1], -1.0, 1.0)
    idx += 2

    # Contacts (left_leg, right_leg, body) (3)
    cond[idx] = float(lander_state.contacts[0])
    cond[idx + 1] = float(lander_state.contacts[1])
    cond[idx + 2] = float(lander_state.contacts[2])
    idx += 3

    # Obstacles: 5 × (dx, dy, r), zero-padded (15)
    for i in range(5):
        if i < len(obstacles):
            cond[idx] = _norm_clip(obstacles[i].dx, *_COND_BOUNDS["obs_dx"])
            cond[idx + 1] = _norm_clip(obstacles[i].dy, *_COND_BOUNDS["obs_dy"])
            cond[idx + 2] = _norm_clip(obstacles[i].r, *_COND_BOUNDS["obs_r"])
        idx += 3

    # Waypoint goals: 5 × 12, zero-padded (60)
    for i in range(5):
        if i < len(waypoint_goals):
            wp = waypoint_goals[i]
            cond[idx] = _norm_clip(wp.t, *_COND_BOUNDS["wp_t"])
            cond[idx + 1] = _norm_clip(wp.t_margin, *_COND_BOUNDS["wp_margin"])
            cond[idx + 2] = _norm_clip(wp.q[0], *_COND_BOUNDS["wp_pos"])
            cond[idx + 3] = _norm_clip(wp.q[1], *_COND_BOUNDS["wp_pos"])
            cond[idx + 4] = _norm_clip(wp.q[2], -math.pi, math.pi)
            cond[idx + 5] = _norm_clip(wp.q_margin[0], *_COND_BOUNDS["wp_margin"])
            cond[idx + 6] = _norm_clip(wp.q_margin[1], *_COND_BOUNDS["wp_margin"])
            cond[idx + 7] = _norm_clip(wp.q_margin[2], *_COND_BOUNDS["wp_margin"])
            cond[idx + 8] = _norm_clip(wp.q_prime[0], *_COND_BOUNDS["wp_vel"])
            cond[idx + 9] = _norm_clip(wp.q_prime[1], *_COND_BOUNDS["wp_vel"])
            cond[idx + 10] = _norm_clip(wp.q_prime[2], *_COND_BOUNDS["wp_vel"])
            cond[idx + 11] = _norm_clip(wp.q_prime_margin[0], *_COND_BOUNDS["wp_margin"])
        idx += 12

    # Guidance actions: 5 × 6, zero-padded (30)
    for i in range(5):
        if i < len(guidance_actions):
            ga = guidance_actions[i]
            cond[idx] = _norm_clip(ga.thrust_v, *_COND_BOUNDS["act_thrust"])
            cond[idx + 1] = _norm_clip(ga.thrust_v_margin, *_COND_BOUNDS["act_margin"])
            cond[idx + 2] = _norm_clip(ga.thrust_h, *_COND_BOUNDS["act_thrust"])
            cond[idx + 3] = _norm_clip(ga.thrust_h_margin, *_COND_BOUNDS["act_margin"])
            cond[idx + 4] = _norm_clip(ga.thrust_t, *_COND_BOUNDS["act_t"])
            cond[idx + 5] = _norm_clip(ga.thrust_t_margin, *_COND_BOUNDS["act_t_margin"])
        idx += 6

    # action_horizon (1)
    cond[idx] = _norm_clip(action_horizon, *_COND_BOUNDS["action_horizon"])
    idx += 1

    # target_frequency (1)
    cond[idx] = _norm_clip(target_frequency, *_COND_BOUNDS["target_freq"])
    idx += 1

    assert idx == STATE_DIM, f"State dims: {idx} vs {STATE_DIM}"

    # ── CFG dims (12) ───────────────────────────────────────────────
    cfg_idx = STATE_DIM

    # waypoint_results: 5 slots (5)
    for i in range(5):
        for cfg in classifier_free_guidance:
            if isinstance(cfg, WaypointResult):
                # Match by index: first WaypointResult → slot 0, etc.
                cond[cfg_idx + i] = float(cfg.outcome) * cfg.alpha
                break
        # Only fill slot 0 from the first WaypointResult, rest stay 0
        if i == 0:
            for cfg in classifier_free_guidance:
                if isinstance(cfg, WaypointResult):
                    cond[cfg_idx] = float(cfg.outcome) * cfg.alpha
                    break
    cfg_idx += 5

    # action_results: 5 slots (5)
    for i in range(5):
        if i == 0:
            for cfg in classifier_free_guidance:
                if isinstance(cfg, ActionResult):
                    cond[cfg_idx] = float(cfg.outcome) * cfg.alpha
                    break
    cfg_idx += 5

    # t_contact (1)
    for cfg in classifier_free_guidance:
        if isinstance(cfg, ContactGuidance):
            cond[cfg_idx] = _norm_clip(cfg.t_contact * cfg.alpha, -10.0, 10.0)
            break
    cfg_idx += 1

    # t_crash (1)
    for cfg in classifier_free_guidance:
        if isinstance(cfg, CrashGuidance):
            cond[cfg_idx] = _norm_clip(cfg.t_crash * cfg.alpha, -10.0, 10.0)
            break
    cfg_idx += 1

    assert cfg_idx == COND_DIM, f"CFG dims: {cfg_idx} vs {COND_DIM}"
    return cond


# ── Obs-space to world-space conversion ─────────────────────────────────────

def _obs_to_world(obs_x: float, obs_y: float) -> tuple[float, float]:
    """Convert observation-space (x, y) to world coords."""
    return (
        obs_x * OBS_X_NORM_SCALE + OBS_X_NORM_OFFSET,
        obs_y * OBS_Y_NORM_SCALE + OBS_Y_NORM_OFFSET,
    )


# ── Main entry point ────────────────────────────────────────────────────────

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
    """Run diffusion model inference and return a ThrustSpline.

    Builds the 131-dim conditioning vector from ICD inputs, runs DDIM
    sampling with classifier-free guidance, and converts the 30-dim
    output (15 B-spline CPs × 2) into a callable spline.
    """
    # If no waypoint goals provided, default to landing pad
    if not waypoint_goals:
        # Convert current lander obs-space position to world frame
        lander_wx, lander_wy = _obs_to_world(
            lander_state.q[0], lander_state.q[1]
        )
        # Waypoint: pad center, relative to lander
        pad_rel_x = PAD_CX - lander_wx  # pad position in lander-relative world coords
        pad_rel_y = PAD_Y - lander_wy
        remaining_t = timeout - lander_state.t_sim_lander
        waypoint_goals = [
            WaypointTarget(
                t=remaining_t,
                t_margin=2.0,
                q=(pad_rel_x, pad_rel_y, 0.0),
                q_margin=(1.0, 1.0, 1.0),
                q_prime=(0.0, -0.5, 0.0),     # slight downward velocity
                q_prime_margin=(2.0, 2.0, 2.0),
            )
        ]

    # If no guidance actions provided, add a neutral one
    if not guidance_actions:
        guidance_actions = [
            ActionTarget(
                thrust_v=0.0,
                thrust_v_margin=2.0,
                thrust_h=0.0,
                thrust_h_margin=2.0,
                thrust_t=t_obs_cmd_latency,
                thrust_t_margin=0.5,
            )
        ]

    # If no CFG provided, add success guidance for first waypoint
    if not classifier_free_guidance:
        classifier_free_guidance = [
            WaypointResult(outcome=1, alpha=1.0),
        ]

    # Build conditioning
    cond = _build_cond(
        t_obs_cmd_latency=t_obs_cmd_latency,
        lander_state=lander_state,
        obstacles=obstacles,
        waypoint_goals=waypoint_goals,
        guidance_actions=guidance_actions,
        classifier_free_guidance=classifier_free_guidance,
        action_horizon=action_horizon,
        target_frequency=target_frequency,
    )

    # Run inference
    sampler, norm_stats = _get_model()
    cond_tensor = torch.tensor(cond, dtype=torch.float32).unsqueeze(0)

    # Build action_boxes (1, n_guidance, 6) for per-step CP projection
    action_boxes = torch.tensor(
        [[ga.thrust_v, ga.thrust_v_margin, ga.thrust_h, ga.thrust_h_margin,
          ga.thrust_t, ga.thrust_t_margin] for ga in guidance_actions],
        dtype=torch.float32,
    ).unsqueeze(0)  # (1, n_guidance, 6)

    x_norm = sampler.sample_cfg(
        cond_tensor,
        guidance_scale=2.0,
        device="cpu",
        norm_stats=norm_stats,
        action_boxes=action_boxes,
        action_horizon=action_horizon,
    )

    # Denormalize control points
    x_mean = torch.tensor(norm_stats["x_mean"])
    x_std = torch.tensor(norm_stats["x_std"])
    x_raw = (x_norm * x_std + x_mean).squeeze(0).numpy()

    # Build spline from model output: (30,) flat → (15, 2) control points
    n_cps = X_DIM // 2
    cps = np.column_stack([x_raw[:n_cps], x_raw[n_cps:]])

    # Pin first CP to current thrust so spline(0) == observed thrust
    cps[0, 0] = float(lander_state.thrust[0])
    cps[0, 1] = float(lander_state.thrust[1])
    spline = _ThrustSplineImpl.from_control_points(cps, t_start=0.0, t_end=action_horizon)

    # Capture guidance actions for post-inference clamping
    _guidance = list(guidance_actions)

    def thrust_fn(t: float) -> ThrustVec:
        tv_val = spline(t)
        v, h = float(np.clip(tv_val.v, -1, 1)), float(np.clip(tv_val.h, -1, 1))
        for at in _guidance:
            if abs(t - at.thrust_t) <= at.thrust_t_margin:
                v = float(np.clip(v, at.thrust_v - at.thrust_v_margin,
                                  at.thrust_v + at.thrust_v_margin))
                h = float(np.clip(h, at.thrust_h - at.thrust_h_margin,
                                  at.thrust_h + at.thrust_h_margin))
        return (v, h)

    return thrust_fn
