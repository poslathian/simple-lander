"""Quick diagnostic: run one PIH episode and print state at key moments.

First measures the actual leg-to-body offset at spring equilibrium (before
any planning) to verify the ~0.44 m actual vs 0.6 m assumed discrepancy.

Usage:
  python scripts/diag_pih.py [--seed N] [--height H]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kto_lander import LanderParams, compound_inertia_about_body_com, DT
from pih_env import (
    PackageInHoleEnv, PIHConfig, TerminationReason,
    PIH_START_X, PIH_PICKUP_X, PIH_PAD_Y, PIH_LEG_OFFSET, PIH_TIMEOUT,
)
from pih_solver import plan_pih_with_kto, PackageInHoleKTOController


def measure_leg_offset(env, n_settle: int = 50) -> None:
    """Print actual leg.worldCenter.y - lander.position.y at spring equilibrium.

    Runs n_settle noop steps so the revolute-joint spring settles, then samples
    both legs.  This is the first thing to print so any geometry mismatch is
    visible before the plan runs.
    """
    noop = np.array([-1.0, 0.0], dtype=np.float32)
    for _ in range(n_settle):
        env.step(noop)
    uw   = env.unwrapped
    body_y = float(uw.lander.position.y)
    offsets = [float(leg.worldCenter.y) - body_y for leg in uw.legs]
    print(f"[leg offset @ equilibrium]  body_y={body_y:.3f}")
    print(f"  left leg : worldCenter.y - body_y = {offsets[0]:+.4f} m")
    print(f"  right leg: worldCenter.y - body_y = {offsets[1]:+.4f} m")
    print(f"  PIH_LEG_OFFSET (assumed) = {PIH_LEG_OFFSET:.4f} m")
    print(f"  discrepancy              = {offsets[0] - (-PIH_LEG_OFFSET):+.4f} m  "
          f"(expect ~+0.16 m at equilibrium)")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",   type=int,   default=70000)
    p.add_argument("--height", type=float, default=2.0,
                   help="package_height_true (default 2.0)")
    args = p.parse_args()

    cfg = PIHConfig(
        hole_depth=1.0,
        package_height_true=args.height,
        package_height_assumed=0.5,
        package_mass_true=2.0,
        package_mass_assumed=2.0,
    )

    print(f"=== PIH diagnostic  seed={args.seed}  h_true={args.height} ===")
    print(f"PIH_START_X={PIH_START_X}  PIH_PICKUP_X={PIH_PICKUP_X}  PIH_PAD_Y={PIH_PAD_Y}")
    print(f"true_contact_lander_y  = {cfg.true_contact_lander_y:.3f} m")
    print(f"assumed_contact_lander_y = {cfg.assumed_contact_lander_y:.3f} m")
    print(f"extraction_lander_y    = {cfg.extraction_lander_y:.3f} m")
    print(f"package_top_y (true)   = {cfg.package_top_y:.3f} m")
    print()

    # ── Leg offset measurement ─────────────────────────────────────────────
    env_settle = PackageInHoleEnv(config=cfg, render_mode=None)
    env_settle.reset(seed=args.seed)
    env_settle.unwrapped.lander.linearVelocity  = (0.0, 0.0)
    env_settle.unwrapped.lander.angularVelocity = 0.0
    measure_leg_offset(env_settle, n_settle=50)
    env_settle.close()

    # ── Plan ─────────────────────────────────────────────────────────────
    env = PackageInHoleEnv(config=cfg, render_mode=None)
    env.reset(seed=args.seed)
    uw = env.unwrapped
    uw.lander.linearVelocity  = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0

    x0 = float(uw.lander.position.x)
    y0 = float(uw.lander.position.y)
    print(f"[start] x={x0:.3f}  y={y0:.3f}")

    mass, inertia = compound_inertia_about_body_com(env)
    params = LanderParams(mass=mass, inertia=inertia)
    print(f"[params] mass={mass:.3f} kg  inertia={inertia:.4f} kg·m²")
    print()

    try:
        plan, T, waypoints = plan_pih_with_kto(x0, y0, 0.0, 0.0, cfg, params, verbose=True)
        kto_feasible = True
    except RuntimeError as exc:
        print(f"[KTO FAILED] {exc}")
        kto_feasible = False

    if not kto_feasible:
        env.close()
        return

    print(f"\n[plan] duration T={T:.2f}s")
    print("[plan] waypoints:")
    for f in dataclasses.fields(waypoints):
        wx, wy, ws = getattr(waypoints, f.name)
        t_abs = ws * T
        ref = plan(t_abs)
        print(f"  {f.name:<14s} ({wx:6.2f}, {wy:5.2f})  "
              f"s={ws:.2f}  t={t_abs:5.1f}s  "
              f"plan→({ref[0]:.2f}, {ref[1]:.2f})")
    print()

    ctrl = PackageInHoleKTOController(plan, waypoints, cfg, params)

    # ── Episode ─────────────────────────────────────────────────────────
    print("--- episode ---")
    done   = False
    step_i = 0
    reason = TerminationReason.NONE

    while not done:
        L    = uw.lander
        t_sim = uw.elapsed_s

        near_pickup = abs(L.position.x - PIH_PICKUP_X) < 2.5 and L.position.y < 9.0
        every_50    = step_i % 50 == 0
        if every_50 or (near_pickup and step_i % 5 == 0):
            leg_ys = [f"{leg.worldCenter.y:.2f}" for leg in uw.legs]
            print(f"  step {step_i:4d} t={t_sim:.1f}s  "
                  f"x={L.position.x:.3f} y={L.position.y:.3f}  "
                  f"vx={L.linearVelocity.x:.2f} vy={L.linearVelocity.y:.2f}  "
                  f"att={uw._attached}  legs_y={leg_ys}")

        action, dbg = ctrl.step(env)
        obs, _r, term, trunc, info = env.step(action)
        done   = term or trunc
        reason = info.get("termination_reason", TerminationReason.NONE)
        step_i += 1

    L = uw.lander
    print(f"\n[done] steps={step_i}  reason={reason.value}")
    print(f"[done] x={L.position.x:.3f} y={L.position.y:.3f}  "
          f"vx={L.linearVelocity.x:.2f} vy={L.linearVelocity.y:.2f}")
    env.close()


if __name__ == "__main__":
    main()
