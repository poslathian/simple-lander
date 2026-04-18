"""Collect PIH episodes with oracle replan for Phase 3 training.

Per-episode record is saved as JSONL (one JSON object per line, append-safe).
Arrays are stored as nested lists; load with json.loads() + np.array().

Actual linear/angular acceleration is computed from velocity differences (Box2D
does not expose acceleration directly): actual_accel[t] = (vel[t+1] - vel[t]) / DT.
The last step uses accel = 0 as a sentinel.

Usage:
  python scripts/collect_pih_phase2.py \\
    --n_episodes 200 \\
    --trigger on_contact \\
    --weld_strategy point \\
    --out results/phase2/collected.jsonl

  # Append-safe: resume from seed S by passing --start_seed S
  python scripts/collect_pih_phase2.py --n_episodes 50 --start_seed 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kto_lander import LanderParams, DT, compound_inertia_about_body_com
from pih_env import PackageInHoleEnv, PIHConfig, TerminationReason
from pih_solver import (
    PIHWaypoints, plan_pih_with_kto, PackageInHoleKTOController,
    _plan_to_numpy,
)


def _collect_episode(
    cfg: PIHConfig,
    seed: int,
    replan_triggers: list[str],
    weld_strategy: str,
    replan_interval_s: float,
    verbose: bool = False,
) -> dict:
    """Run one PIH episode, return the complete data record."""
    env = PackageInHoleEnv(config=cfg, render_mode=None)
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped

    x0 = float(uw.lander.position.x)
    y0 = float(uw.lander.position.y)

    mass, inertia = compound_inertia_about_body_com(env)
    params = LanderParams(mass=mass, inertia=inertia)

    # Plan (assumed parameters)
    kto_feasible = True
    plan = waypoints = None
    try:
        plan, plan_T, waypoints = plan_pih_with_kto(
            x0, y0, 0.0, 0.0, cfg, params, verbose=verbose,
        )
    except RuntimeError as exc:
        kto_feasible = False
        if verbose:
            print(f"  seed={seed}: KTO FAILED — {exc}")

    if not kto_feasible:
        env.close()
        return {
            "seed": seed,
            "cfg":  _cfg_to_dict(cfg),
            "kto_plan": {"feasible": False, "cps": None, "knots": None, "waypoints": None},
            "replan_events": [],
            "states": [], "actions": [], "actual_accel": [], "timestamps": [],
            "attachment_time": None, "attachment_state": None,
            "termination_reason": TerminationReason.TIMEOUT.value,
            "total_reward": 0.0,
            "episode_length_s": 0.0,
        }

    # Extract KTO plan arrays for the record
    kto_cps, kto_knots = _plan_to_numpy(plan)
    kto_plan_record = {
        "feasible":  True,
        "cps":       kto_cps.tolist(),
        "knots":     kto_knots.tolist(),
        "waypoints": waypoints.to_dict(),
    }

    ctrl = PackageInHoleKTOController(
        plan, waypoints, cfg, params,
        replan_triggers=replan_triggers,
        replan_interval_s=replan_interval_s,
        weld_strategy=weld_strategy,
    )

    # Episode loop
    states:       list[list[float]] = []
    actions:      list[list[float]] = []
    timestamps:   list[float]       = []
    vel_history:  list[list[float]] = []  # for accel computation
    total_reward  = 0.0
    attachment_time:  float | None = None
    attachment_state: list | None  = None

    done = False
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

        # Attachment event
        if uw._attached and attachment_time is None:
            attachment_time = ctrl.t
            attachment_state = state[:]

        action, _ = ctrl.step(env)
        actions.append(action.tolist())

        _, reward, term, trunc, info = env.step(action)
        total_reward += float(reward)
        done = term or trunc

    reason = info.get("termination_reason", TerminationReason.NONE)
    episode_length_s = ctrl.t

    # Compute actual linear + angular acceleration from velocity differences
    # actual_accel[t] = (vel[t+1] - vel[t]) / DT;  last step = zero sentinel
    actual_accel: list[list[float]] = []
    for i in range(len(vel_history)):
        if i + 1 < len(vel_history):
            v0 = vel_history[i]
            v1 = vel_history[i + 1]
            ax = (v1[0] - v0[0]) / DT
            ay = (v1[1] - v0[1]) / DT
            al = (v1[2] - v0[2]) / DT
        else:
            ax, ay, al = 0.0, 0.0, 0.0
        actual_accel.append([ax, ay, al])

    env.close()

    return {
        "seed": seed,
        "cfg":  _cfg_to_dict(cfg),

        "states":       states,
        "actions":      actions,
        "actual_accel": actual_accel,
        "timestamps":   timestamps,

        "kto_plan":     kto_plan_record,

        "replan_events": ctrl.replan_events,

        "attachment_time":  attachment_time,
        "attachment_state": attachment_state,

        "termination_reason": reason.value,
        "total_reward":       total_reward,
        "episode_length_s":   episode_length_s,

        "wp_idx_at_end": ctrl.wp_idx,
    }


def _cfg_to_dict(cfg: PIHConfig) -> dict:
    return {f.name: getattr(cfg, f.name) for f in __import__("dataclasses").fields(cfg)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes",       type=int,   default=200)
    ap.add_argument("--start_seed",       type=int,   default=80000,
                    help="First seed (default 80000, outside Phase 1 validation seeds)")
    ap.add_argument("--trigger",          type=str,   default="on_contact",
                    help="Comma-separated triggers: on_contact,periodic (or 'none')")
    ap.add_argument("--weld_strategy",    type=str,   default="point")
    ap.add_argument("--replan_interval",  type=float, default=5.0)
    ap.add_argument("--hole_depth",       type=float, default=1.0)
    ap.add_argument("--pkg_height_true",  type=float, default=1.5)
    ap.add_argument("--pkg_height_assumed", type=float, default=0.5)
    ap.add_argument("--pkg_mass_true",    type=float, default=2.0)
    ap.add_argument("--pkg_mass_assumed", type=float, default=2.0)
    ap.add_argument("--out",              type=str,   default="results/phase2/collected.jsonl")
    ap.add_argument("--verbose",          action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    triggers = [] if args.trigger == "none" else args.trigger.split(",")

    cfg = PIHConfig(
        hole_depth=args.hole_depth,
        package_height_true=args.pkg_height_true,
        package_height_assumed=args.pkg_height_assumed,
        package_mass_true=args.pkg_mass_true,
        package_mass_assumed=args.pkg_mass_assumed,
        start_at_pad=True,
    )

    print(f"Collecting {args.n_episodes} episodes  trigger={triggers}  "
          f"strategy={args.weld_strategy}  → {out_path}")
    print(f"  cfg: h_true={cfg.package_height_true}  h_assumed={cfg.package_height_assumed}  "
          f"m_true={cfg.package_mass_true}  m_assumed={cfg.package_mass_assumed}")

    # Count already-collected episodes (for resume)
    n_existing = 0
    if out_path.exists():
        with open(out_path) as f:
            n_existing = sum(1 for line in f if line.strip())
        print(f"  Resuming: {n_existing} episodes already in file")

    t_start = time.monotonic()
    n_done  = 0
    n_success = 0
    n_replan  = 0

    with open(out_path, "a") as f_out:
        for i in range(args.n_episodes):
            seed = args.start_seed + n_existing + i
            t0 = time.monotonic()
            record = _collect_episode(
                cfg, seed, triggers, args.weld_strategy, args.replan_interval,
                verbose=args.verbose,
            )
            elapsed = time.monotonic() - t0

            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            n_done  += 1
            if record["termination_reason"] == TerminationReason.SUCCESS.value:
                n_success += 1
            n_replan += len(record["replan_events"])

            print(f"  ep {n_existing + i + 1:4d}  seed={seed}  "
                  f"reason={record['termination_reason']:20s}  "
                  f"replans={len(record['replan_events'])}  "
                  f"T={record['episode_length_s']:5.1f}s  "
                  f"({elapsed:.1f}s wall)")

    total_wall = time.monotonic() - t_start
    print(f"\nDone: {n_done} episodes  success={n_success}/{n_done}  "
          f"total_replans={n_replan}  wall={total_wall:.1f}s")


if __name__ == "__main__":
    main()
