"""Render one PIH episode and save frames to results/pih_render/.

Sets SDL_VIDEODRIVER=dummy so no display is needed (works in Docker on Windows).

Usage:
  python scripts/render_pih.py [--seed N] [--every N] [--raycasts]
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from kto_lander import LanderParams, compound_inertia_about_body_com
from pih_env import PackageInHoleEnv, PIHConfig, TerminationReason
from pih_solver import plan_pih_with_kto, PackageInHoleKTOController


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",              type=int,   default=70000)
    p.add_argument("--every",             type=int,   default=10,
                   help="save one frame every N steps (default 10 → 5 fps equivalent)")
    p.add_argument("--raycasts",          action="store_true")
    p.add_argument("--replan",            action="store_true",
                   help="enable on_contact oracle replan")
    p.add_argument("--pkg_height_true",   type=float, default=1.5)
    p.add_argument("--pkg_height_assumed",type=float, default=0.5)
    p.add_argument("--pkg_mass_true",     type=float, default=2.0)
    p.add_argument("--pkg_mass_assumed",  type=float, default=2.0)
    args = p.parse_args()

    out_dir = Path("results/pih_render")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PIHConfig(
        hole_depth=1.0,
        package_height_true=args.pkg_height_true,
        package_height_assumed=args.pkg_height_assumed,
        package_mass_true=args.pkg_mass_true,
        package_mass_assumed=args.pkg_mass_assumed,
        start_at_pad=True,
    )

    env = PackageInHoleEnv(config=cfg, render_mode="rgb_array")
    env.reset(seed=args.seed)
    uw = env.unwrapped
    if args.raycasts:
        uw._show_raycasts = True

    x0 = float(uw.lander.position.x)
    y0 = float(uw.lander.position.y)

    mass, inertia = compound_inertia_about_body_com(env)
    params = LanderParams(mass=mass, inertia=inertia)

    triggers = ["on_contact"] if args.replan else []
    print(f"[render] planning for seed={args.seed}  replan={args.replan} ...")
    plan, T, waypoints = plan_pih_with_kto(x0, y0, 0.0, 0.0, cfg, params, verbose=True)
    ctrl = PackageInHoleKTOController(plan, waypoints, cfg, params,
                                      replan_triggers=triggers)
    print(f"[render] plan T={T:.1f}s  saving every {args.every} steps → {out_dir}")

    done = False
    step_i = 0
    n_saved = 0

    while not done:
        if step_i % args.every == 0:
            frame = env.render()
            if frame is not None:
                plt.imsave(out_dir / f"frame_{step_i:05d}.png", frame)
                n_saved += 1

        action, _ = ctrl.step(env)
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
        step_i += 1

    reason = info.get("termination_reason", TerminationReason.NONE)

    # Hold the final frame; longer pause for collision events so the marker is visible.
    hold = 50 if reason == TerminationReason.EXTRACTION_COLLISION else 5
    frame = env.render()
    if frame is not None:
        for h in range(hold):
            plt.imsave(out_dir / f"frame_{step_i + h:05d}.png", frame)
            n_saved += 1

    print(f"[render] done  steps={step_i}  reason={reason.value}  saved={n_saved} frames")
    env.close()


if __name__ == "__main__":
    main()
