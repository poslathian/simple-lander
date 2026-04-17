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
    p.add_argument("--seed",     type=int,  default=70000)
    p.add_argument("--every",    type=int,  default=10,
                   help="save one frame every N steps (default 10 → 5 fps equivalent)")
    p.add_argument("--raycasts", action="store_true")
    args = p.parse_args()

    out_dir = Path("results/pih_render")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PIHConfig(
        hole_depth=1.0,
        package_height_true=1.5,
        package_height_assumed=0.5,
        package_mass_true=2.0,
        package_mass_assumed=2.0,
    )

    env = PackageInHoleEnv(config=cfg, render_mode="rgb_array")
    env.reset(seed=args.seed)
    uw = env.unwrapped
    uw.lander.linearVelocity  = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0
    if args.raycasts:
        uw._show_raycasts = True

    x0 = float(uw.lander.position.x)
    y0 = float(uw.lander.position.y)

    mass, inertia = compound_inertia_about_body_com(env)
    params = LanderParams(mass=mass, inertia=inertia)

    print(f"[render] planning for seed={args.seed} ...")
    plan, T, waypoints = plan_pih_with_kto(x0, y0, 0.0, 0.0, cfg, params, verbose=True)
    ctrl = PackageInHoleKTOController(plan, waypoints, cfg, params)
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

    frame = env.render()
    if frame is not None:
        plt.imsave(out_dir / f"frame_{step_i:05d}_final.png", frame)
        n_saved += 1

    reason = info.get("termination_reason", TerminationReason.NONE)
    print(f"[render] done  steps={step_i}  reason={reason.value}  saved={n_saved} frames")
    env.close()


if __name__ == "__main__":
    main()
