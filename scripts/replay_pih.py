"""Corrected-plan replay for PIH Phase 2.5 (Component 3).

replay_with_correction():
  Resets the env with the original seed, follows the original KTO plan
  until the previous named waypoint is crossed, then direct-switches to
  the user-corrected plan.  Returns a full episode record in the Phase 2
  data format with correction_type='user_waypoint'.

No Drake required — both plans are reconstructed from saved CPS/knots
via scipy (pih_solver._numpy_to_plan), so this can run locally.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kto_lander import LanderParams, DT, compound_inertia_about_body_com
from pih_env import PackageInHoleEnv, PIHConfig, TerminationReason
from pih_solver import (
    PIHWaypoints, PackageInHoleKTOController,
    _numpy_to_plan,
)


def replay_with_correction(
    seed: int,
    cfg: PIHConfig,
    original_cps: np.ndarray,
    original_knots: np.ndarray,
    original_waypoints_dict: dict,
    corrected_cps: np.ndarray,
    corrected_knots: np.ndarray,
    corrected_waypoints_dict: dict,
    previous_waypoint_name: str,
    user_waypoint: tuple[float, float],
    frame_index: int,
    verbose: bool = False,
) -> dict:
    """Replay episode with corrected plan spliced at previous_waypoint_name.

    Phase 1  Original plan controller runs until previous_waypoint_name is
             crossed (same actions as the original failure episode, by Box2D
             determinism).
    Phase 2  Controller switches directly to the corrected plan. Trajectory
             diverges here — success/failure depends on correction quality.

    Returns a full episode record (Phase 2 format) with
    correction_type='user_waypoint' and a replan_events entry describing
    the correction.
    """
    env = PackageInHoleEnv(config=cfg, render_mode=None)
    env.reset(seed=seed)
    uw = env.unwrapped

    mass, inertia = compound_inertia_about_body_com(env)
    params = LanderParams(mass=mass, inertia=inertia)

    original_plan   = _numpy_to_plan(original_cps, original_knots)
    corrected_plan  = _numpy_to_plan(corrected_cps, corrected_knots)

    def _wpts_from_dict(d: dict) -> PIHWaypoints:
        return PIHWaypoints(**{
            k: tuple(float(v) for v in vals)
            for k, vals in d.items()
        })

    original_wpts  = _wpts_from_dict(original_waypoints_dict)
    corrected_wpts = _wpts_from_dict(corrected_waypoints_dict)

    # ── Phase 1: original plan until previous_waypoint_name is crossed ───
    ctrl = PackageInHoleKTOController(
        original_plan, original_wpts,
        cfg, params, replan_triggers=[],
    )

    states:      list[list[float]] = []
    actions:     list[list[float]] = []
    timestamps:  list[float]       = []
    vel_history: list[list[float]] = []
    total_reward = 0.0
    attachment_time:  Optional[float]      = None
    attachment_state: Optional[list[float]] = None
    splice_time:  Optional[float]      = None
    splice_state: Optional[list[float]] = None

    done = False
    switched = False

    while not done:
        lander = uw.lander
        pos = lander.position
        vel = lander.linearVelocity
        state = [
            float(pos.x), float(pos.y),
            float(vel.x), float(vel.y),
            float(lander.angle), float(lander.angularVelocity),
        ]
        states.append(state)
        timestamps.append(ctrl.t)
        vel_history.append([float(vel.x), float(vel.y), float(lander.angularVelocity)])

        if uw._attached and attachment_time is None:
            attachment_time = ctrl.t
            attachment_state = state[:]

        # Switch to corrected plan when previous_waypoint_name is first crossed
        if not switched and previous_waypoint_name in ctrl.waypoint_states:
            switched = True
            splice_time  = ctrl.t
            splice_state = state[:]
            if verbose:
                print(f"[replay] splice at t={splice_time:.2f}s  "
                      f"prev_wp={previous_waypoint_name}  "
                      f"pos=({state[0]:.2f},{state[1]:.2f})")

            # Build corrected controller. It will detect uw._attached on its
            # first step and update mass automatically (same logic as original).
            ctrl = PackageInHoleKTOController(
                corrected_plan, corrected_wpts,
                cfg, params, replan_triggers=[],
            )

        action, _ = ctrl.step(env)
        actions.append(action.tolist())

        _, reward, term, trunc, info = env.step(action)
        total_reward += float(reward)
        done = term or trunc

    reason = info.get("termination_reason", TerminationReason.NONE)
    episode_length_s = ctrl.t if switched else ctrl.t

    if verbose:
        spliced = "YES" if switched else "NO (never reached prev waypoint)"
        print(f"[replay] done  reason={reason.value}  "
              f"spliced={spliced}  T={episode_length_s:.1f}s")

    # Acceleration from velocity differences (same as collect_pih_phase2)
    actual_accel: list[list[float]] = []
    for i in range(len(vel_history)):
        if i + 1 < len(vel_history):
            v0, v1 = vel_history[i], vel_history[i + 1]
            actual_accel.append([
                (v1[0] - v0[0]) / DT,
                (v1[1] - v0[1]) / DT,
                (v1[2] - v0[2]) / DT,
            ])
        else:
            actual_accel.append([0.0, 0.0, 0.0])

    env.close()

    replan_event = {
        "trigger":          "user_waypoint",
        "sim_time":         splice_time,
        "state_at_trigger": splice_state,
        "user_waypoint": {
            "position":               list(user_waypoint),
            "previous_waypoint_name": previous_waypoint_name,
            "frame_index":            frame_index,
        },
        "waypoint_replan": {
            "cps":       corrected_cps.tolist(),
            "knots":     corrected_knots.tolist(),
            "waypoints": corrected_waypoints_dict,
            "feasible":  True,
        },
        "welded_plan": None,  # direct switch; no weld needed (start matches splice state)
    }

    return {
        "seed":            seed,
        "cfg":             {f.name: getattr(cfg, f.name)
                            for f in dataclasses.fields(cfg)},
        "correction_type": "user_waypoint",

        "states":       states,
        "actions":      actions,
        "actual_accel": actual_accel,
        "timestamps":   timestamps,

        "kto_plan": {
            "feasible": True,
            "cps":      original_cps.tolist(),
            "knots":    original_knots.tolist(),
        },

        "replan_events":   [replan_event] if switched else [],
        "waypoint_states": ctrl.waypoint_states,

        "attachment_time":  attachment_time,
        "attachment_state": attachment_state,

        "termination_reason": reason.value,
        "total_reward":       total_reward,
        "episode_length_s":   episode_length_s,
        "splice_happened":    switched,
    }


