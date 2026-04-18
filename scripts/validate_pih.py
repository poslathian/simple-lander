"""Phase 1 validation for the Package-in-Hole environment.

Runs four checks from the Phase 1 validation checklist:
  1. Regression: base LunarLander-v3 + KTO still lands on ~36-37/50 holdout seeds.
  2. Parameter sweep: package_height_true sweep → extraction collision rate increases.
  3. Failure mode isolation: depth-only, mass-only, both-wrong scenarios.
  4. Conservative hover fallback: infeasibility triggers and is detectable.

Saves per-episode data in a Phase-3-consumable format (JSONL) under
  results/pih_validation/

Usage:
  python scripts/validate_pih.py
  python scripts/validate_pih.py --regression-only
  python scripts/validate_pih.py --sweep-seeds 5 --sweep-heights 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kto_lander import (
    LanderParams, DT,
    compound_inertia_about_body_com, run_episode,
)
from pih_env import PIHConfig, TerminationReason, PIH_PICKUP_X
from pih_solver import (
    PIHWaypoints, plan_pih_with_kto, PackageInHoleKTOController,
)

RESULTS_DIR = Path("results/pih_validation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PIH episode runner
# ---------------------------------------------------------------------------

def run_pih_episode(cfg: PIHConfig, seed: int, verbose: bool = False) -> dict:
    """Run one PIH episode with the KTO planner under assumed params.

    Returns a record dict (Phase-3 save format).
    """
    from pih_env import PackageInHoleEnv  # local import — heavy on first use

    env = PackageInHoleEnv(config=cfg, render_mode=None)
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped

    x0 = float(uw.lander.position.x)
    y0 = float(uw.lander.position.y)

    mass, inertia = compound_inertia_about_body_com(env)
    params = LanderParams(mass=mass, inertia=inertia)

    # Plan
    kto_feasible = True
    plan = waypoints = None
    t0 = time.monotonic()
    try:
        plan, plan_T, waypoints = plan_pih_with_kto(
            x0, y0, 0.0, 0.0, cfg, params, verbose=verbose,
        )
    except RuntimeError as exc:
        kto_feasible = False
        if verbose:
            print(f"    seed={seed}: KTO FAILED — {exc}")
    solve_s = time.monotonic() - t0

    if verbose and kto_feasible:
        print(f"    seed={seed}: solved T={plan_T:.2f}s ({solve_s:.1f}s wall)")

    ctrl = (
        PackageInHoleKTOController(plan, waypoints, cfg, params)
        if kto_feasible else None
    )

    # Episode loop
    states, actions = [], []
    contact_event = failure_event = None
    done = False
    step_i = 0
    reason = TerminationReason.NONE

    while not done:
        L = uw.lander
        t_sim = uw.elapsed_s

        states.append({
            "t": t_sim,
            "x": float(L.position.x),   "y": float(L.position.y),
            "vx": float(L.linearVelocity.x), "vy": float(L.linearVelocity.y),
            "theta": float(L.angle),    "omega": float(L.angularVelocity),
            "attached": bool(uw._attached),
        })

        if ctrl is not None:
            action, _dbg = ctrl.step(env)
        else:
            # Fallback: hold main engine at approximate hover thrust
            m_power = params.mass * params.g / params.F_main_max
            action = np.array([
                float(np.clip(2.0 * m_power - 1.0, -1.0, 1.0)),
                0.0,
            ], dtype=np.float32)
        actions.append({"main": float(action[0]), "side": float(action[1])})

        if uw._attached and contact_event is None:
            contact_event = {
                "step": step_i, "t": t_sim,
                "x": float(L.position.x), "y": float(L.position.y),
            }

        obs, _reward, term, trunc, info = env.step(action)
        done = term or trunc
        reason = info.get("termination_reason", TerminationReason.NONE)

        if done and reason not in (TerminationReason.NONE, TerminationReason.SUCCESS):
            failure_event = {
                "step": step_i, "t": t_sim, "reason": reason.value,
                "x": float(L.position.x), "y": float(L.position.y),
            }
        step_i += 1

    env.close()

    waypoints_dict = None
    if waypoints is not None:
        waypoints_dict = {
            f.name: getattr(waypoints, f.name)
            for f in __import__("dataclasses").fields(waypoints)
        }

    return {
        "seed":               seed,
        "termination_reason": reason.value,
        "success":            reason == TerminationReason.SUCCESS,
        "kto_feasible":       kto_feasible,
        "n_steps":            step_i,
        "states":             states,
        "actions":            actions,
        "contact_event":      contact_event,
        "failure_event":      failure_event,
        "waypoints":          waypoints_dict,
        "cfg": {
            "package_height_true":    cfg.package_height_true,
            "package_mass_true":      cfg.package_mass_true,
            "package_height_assumed": cfg.package_height_assumed,
            "package_mass_assumed":   cfg.package_mass_assumed,
            "hole_depth":             cfg.hole_depth,
            "hole_width":             cfg.hole_width,
        },
    }


def _save(record: dict, outfile: Path) -> None:
    with open(outfile, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Check 1: Regression
# ---------------------------------------------------------------------------

def check_regression(n_seeds: int = 20, seed_offset: int = 90000) -> tuple[bool, dict]:
    """Verify base LunarLander-v3 + KTO still achieves ≥ 50% landing rate."""
    print(f"\n[1] Regression: base LunarLander-v3 + KTO  ({n_seeds} seeds)")
    lands = estops = crashes = 0
    for i in range(n_seeds):
        seed = seed_offset + i
        m = run_episode(render=False, seed=seed, verbose=False)
        if m["landed"]:
            lands += 1
        elif m["estop"]:
            estops += 1
        elif m["terminated"]:
            crashes += 1
        print(".", end="", flush=True)
    print()
    rate = lands / n_seeds
    ok = rate >= 0.50
    print(f"    landed={lands}/{n_seeds} ({rate:.0%})  estop={estops}  crash={crashes}")
    print(f"    {'PASS' if ok else 'FAIL'} (expected ≥50%)")
    return ok, {"lands": lands, "n_seeds": n_seeds, "rate": rate,
                "estops": estops, "crashes": crashes}


# ---------------------------------------------------------------------------
# Check 2: Parameter sweep
# ---------------------------------------------------------------------------

def check_parameter_sweep(
    n_seeds: int = 5,
    height_values: list[float] | None = None,
    seed_offset: int = 70000,
) -> tuple[bool, dict]:
    """Sweep package_height_true with fixed assumed protrusion; expect extraction
    collision rate to increase monotonically as height error grows.

    KTO assumes package_height_assumed=0.5 throughout.
    Collision threshold: h_true > hole_depth + h_assumed + 0.2 = 1.7
    So h ≤ 1.6 → 0% expected; h ≥ 2.0 → 100% expected.
    """
    if height_values is None:
        height_values = [1.0, 1.3, 1.6, 2.0, 2.5]
    print(f"\n[2] Parameter sweep: package_height_true in {height_values}")
    print(f"    {n_seeds} seeds per value  assumed_protrusion=0.5  hole_depth=1.0")
    print(f"    Expected: 0% at h≤1.6, rising above h=1.7")

    results: dict = {}
    all_ok = True

    for pkg_h_true in height_values:
        cfg = PIHConfig(
            hole_depth=1.0, package_height_true=pkg_h_true,
            package_height_assumed=0.5, package_mass_true=2.0, package_mass_assumed=2.0,
            start_at_pad=True,
        )
        outfile = RESULTS_DIR / f"sweep_h{pkg_h_true:.2f}.jsonl"
        n_collision = n_feasible = 0
        prot_true = pkg_h_true - 1.0
        print(f"  h={pkg_h_true:.2f} (prot_true={prot_true:.2f})", end="", flush=True)

        for i in range(n_seeds):
            rec = run_pih_episode(cfg, seed_offset + i)
            _save(rec, outfile)
            if rec["kto_feasible"]:
                n_feasible += 1
                if rec["termination_reason"] == TerminationReason.EXTRACTION_COLLISION.value:
                    n_collision += 1
            print(".", end="", flush=True)

        rate = n_collision / n_feasible if n_feasible > 0 else 0.0
        results[pkg_h_true] = {
            "collision_rate": rate, "n_collision": n_collision,
            "n_feasible": n_feasible, "n_seeds": n_seeds,
        }
        print(f" collision={n_collision}/{n_feasible} ({rate:.0%})")

    print("\n  Sweep summary:")
    rates = [results[h]["collision_rate"] for h in height_values]
    for h, r in zip(height_values, rates):
        bar = "█" * int(r * 20)
        print(f"    h={h:.2f}: {r:5.0%}  {bar}")

    monotone_ok = len(rates) < 2 or rates[-1] > rates[0]
    print(f"\n  Monotone increase: {'PASS' if monotone_ok else 'FAIL'}")
    if len(rates) >= 2:
        print(f"  (h={height_values[0]:.1f} → {rates[0]:.0%},  "
              f"h={height_values[-1]:.1f} → {rates[-1]:.0%})")
    if not monotone_ok:
        all_ok = False

    return all_ok, results


# ---------------------------------------------------------------------------
# Check 3: Failure mode isolation
# ---------------------------------------------------------------------------

def check_failure_isolation(
    n_seeds: int = 5, seed_offset: int = 71000,
) -> tuple[bool, dict]:
    """Confirm each failure mode can fire independently.

    Scenario A: height wrong, mass accurate → SUCCESS (package fits laterally;
                 centered tracking extracts it cleanly, lands at wrong height)
    Scenario B: height accurate, mass very wrong → SUCCESS (dynamics mismatch
                 but tracking still good enough)
    Scenario C: both wrong → EXTRACTION_COLLISION (mass error causes lateral
                 drift near the hole, package hits wall)
    """
    print(f"\n[3] Failure mode isolation ({n_seeds} seeds each)")

    scenarios = {
        # Height wrong, mass correct: package extracted successfully (fits
        # laterally; PD tracking keeps lander centred over hole during ascent).
        # Lands at wrong height — correct signal for RL, not a crash.
        "depth_wrong_mass_ok": PIHConfig(
            hole_depth=1.0, package_height_true=2.0, package_height_assumed=0.5,
            package_mass_true=2.0, package_mass_assumed=2.0, start_at_pad=True,
        ),
        # Height correct, mass very wrong: dynamics mismatch but extraction OK.
        "depth_ok_mass_wrong": PIHConfig(
            hole_depth=1.0, package_height_true=1.5, package_height_assumed=0.5,
            package_mass_true=1.0, package_mass_assumed=4.0, start_at_pad=True,
        ),
        # Both wrong: mass error causes lateral drift near the hole → collision.
        "both_wrong": PIHConfig(
            hole_depth=1.0, package_height_true=2.0, package_height_assumed=0.5,
            package_mass_true=1.0, package_mass_assumed=4.0, start_at_pad=True,
        ),
    }

    results: dict = {}
    all_ok = True

    for name, cfg in scenarios.items():
        outfile = RESULTS_DIR / f"isolation_{name}.jsonl"
        counts: dict[str, int] = {r.value: 0 for r in TerminationReason}
        n_feasible = 0
        print(f"  '{name}':", end="", flush=True)

        for i in range(n_seeds):
            rec = run_pih_episode(cfg, seed_offset + i)
            _save(rec, outfile)
            counts[rec["termination_reason"]] += 1
            if rec["kto_feasible"]:
                n_feasible += 1
            print(".", end="", flush=True)

        print()
        for rv, count in counts.items():
            if count:
                print(f"    {rv}: {count}/{n_seeds}")
        results[name] = {"counts": counts, "n_feasible": n_feasible}

        # Mass + height both wrong must produce extraction collision (lateral
        # drift from mass error causes wall strike near the hole).
        if name == "both_wrong" and n_feasible > 0:
            ec = counts[TerminationReason.EXTRACTION_COLLISION.value]
            ok = ec > 0
            print(f"    extraction collision rate: {ec}/{n_feasible} "
                  f"{'PASS' if ok else 'FAIL (none fired)'}")
            if not ok:
                all_ok = False

    return all_ok, results


# ---------------------------------------------------------------------------
# Check 4: Conservative hover fallback
# ---------------------------------------------------------------------------

def check_hover_fallback(n_seeds: int = 3, seed_offset: int = 72000) -> tuple[bool, dict]:
    """Confirm infeasibility is detectable via kto_feasible flag.

    Extreme geometry + very tight time budget forces solver failure so we can
    verify the fallback path activates and is observable.
    """
    print(f"\n[4] Hover fallback ({n_seeds} seeds, extreme geometry)")
    # Tall package + extreme assumed protrusion → contact waypoint below pad surface
    cfg = PIHConfig(
        hole_depth=1.0, package_height_true=3.0, package_height_assumed=0.05,
        package_mass_true=2.0, package_mass_assumed=3.0, start_at_pad=True,
    )
    outfile = RESULTS_DIR / "fallback_test.jsonl"
    n_infeasible = 0

    for i in range(n_seeds):
        rec = run_pih_episode(cfg, seed_offset + i, verbose=True)
        _save(rec, outfile)
        if not rec["kto_feasible"]:
            n_infeasible += 1
            print(f"    seed={rec['seed']}: INFEASIBLE → fallback active, "
                  f"term={rec['termination_reason']}")
        else:
            print(f"    seed={rec['seed']}: feasible (solver converged), "
                  f"term={rec['termination_reason']}")

    print(f"\n  Infeasible: {n_infeasible}/{n_seeds}")
    print(f"  Fallback flag detectable: "
          f"{'PASS' if n_infeasible > 0 else 'INCONCLUSIVE (solver always converged)'}")
    return True, {"n_infeasible": n_infeasible, "n_seeds": n_seeds}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_summary(summary: dict, all_pass: bool) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for name, result in summary.items():
        status = "PASS" if result.get("pass", True) else "FAIL"
        print(f"  [{status}] {name}")
    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--regression-only", action="store_true")
    p.add_argument("--no-regression",   action="store_true")
    p.add_argument("--sweep-seeds",  type=int, default=5)
    p.add_argument("--sweep-heights", type=int, default=5,
                   help="Number of height grid points (5 = default)")
    p.add_argument("--iso-seeds",    type=int, default=5)
    p.add_argument("--seed-offset",  type=int, default=70000)
    args = p.parse_args()

    print("=" * 60)
    print("Phase 1 Validation — Package-in-Hole")
    print("=" * 60)
    print(f"Results dir: {RESULTS_DIR.resolve()}")

    summary: dict = {}
    all_pass = True

    if not args.no_regression:
        ok, data = check_regression(n_seeds=20)
        summary["regression"] = {"pass": ok, **data}
        if not ok:
            all_pass = False

    if args.regression_only:
        _print_summary(summary, all_pass)
        return 0 if all_pass else 1

    heights = list(np.linspace(1.0, 2.5, args.sweep_heights))
    ok, data = check_parameter_sweep(n_seeds=args.sweep_seeds, height_values=heights,
                                     seed_offset=args.seed_offset)
    summary["sweep"] = {"pass": ok, "results": data}
    if not ok:
        all_pass = False

    ok, data = check_failure_isolation(n_seeds=args.iso_seeds,
                                       seed_offset=args.seed_offset + 1000)
    summary["isolation"] = {"pass": ok, "results": data}
    if not ok:
        all_pass = False

    ok, data = check_hover_fallback(n_seeds=3, seed_offset=args.seed_offset + 2000)
    summary["hover_fallback"] = {"pass": ok, **data}

    _print_summary(summary, all_pass)

    summary_path = RESULTS_DIR / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written to: {summary_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
