"""Standalone C2 weld verification: 100 random splines × 3 strategies.

All 300 tests must pass within float64 tolerance before any pih_solver integration.

Usage:
  python scripts/test_weld.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, str(Path(__file__).parent.parent))
from pih_weld import weld_c2, _make_uniform_clamped_knots


STRATEGIES = ["point", "overlap_1", "overlap_2"]
N_CONSTRAINED = {"point": 3, "overlap_1": 4, "overlap_2": 5}
TOL = 1e-10


def random_clamped_spline(
    n_cps: int, d: int, delta: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random N-CP clamped cubic B-spline; returns (cps, knots)."""
    knots = _make_uniform_clamped_knots(n_cps, delta)
    cps = rng.standard_normal((n_cps, d))
    return cps, knots


def check_c2_at_weld(
    old_cps: np.ndarray,
    old_knots: np.ndarray,
    new_cps: np.ndarray,
    new_knots: np.ndarray,
    weld_param: float,
    tol: float = TOL,
) -> dict[str, float]:
    """Verify C2 continuity using scipy evaluation (WRITEUP.md §Verification).

    Returns dict of {order_name: max_error}.
    Raises AssertionError if any error exceeds tol.
    """
    spl_old = BSpline(old_knots, old_cps, 3)
    spl_new = BSpline(new_knots, new_cps, 3)

    errs: dict[str, float] = {}
    for order, name in ((0, "C0_position"), (1, "C1_velocity"), (2, "C2_acceleration")):
        if order == 0:
            val_old = spl_old(weld_param)
            val_new = spl_new(0.0)
        else:
            val_old = spl_old.derivative(order)(weld_param)
            val_new = spl_new.derivative(order)(0.0)
        err = float(np.max(np.abs(val_old - val_new)))
        errs[name] = err
        if err > tol:
            raise AssertionError(
                f"{name}: error {err:.3e} > tol {tol:.3e}  "
                f"(weld_param={weld_param:.4f})"
            )
    return errs


def run_tests(
    n_trials: int = 100,
    n_cps: int = 20,
    d: int = 2,
    tol: float = TOL,
    seed: int = 42,
    verbose: bool = False,
) -> bool:
    rng = np.random.default_rng(seed)
    total = n_trials * len(STRATEGIES)
    passed = 0
    max_errs: dict[str, float] = {}

    for trial in range(n_trials):
        delta_old = float(rng.uniform(0.5, 2.0))
        old_cps, old_knots = random_clamped_spline(n_cps, d, delta_old, rng)
        T_old = float(old_knots[-1])

        for strategy in STRATEGIES:
            k = N_CONSTRAINED[strategy]
            n_free = n_cps - k
            delta_new = float(rng.uniform(0.5, 2.0))

            # Leave room for overlap strategies to look ahead on old spline
            look_ahead = 2.0 * delta_new + 0.05
            max_weld = T_old - look_ahead
            if max_weld < 0.05:
                # Old spline too short for look-ahead; shorten delta_new
                delta_new = (T_old - 0.1) / 2.1
                max_weld = T_old - 2.0 * delta_new - 0.05
            weld_param = float(rng.uniform(0.0, max(0.01, max_weld)))

            free_cps = rng.standard_normal((n_free, d))

            try:
                new_cps, new_knots = weld_c2(
                    old_cps, old_knots, free_cps, delta_new, weld_param,
                    strategy=strategy,
                )
                errs = check_c2_at_weld(
                    old_cps, old_knots, new_cps, new_knots, weld_param, tol=tol
                )
                for name, err in errs.items():
                    max_errs[name] = max(max_errs.get(name, 0.0), err)
                passed += 1
                if verbose:
                    print(
                        f"  trial={trial:3d} {strategy:<10s} "
                        f"weld={weld_param:.3f}  "
                        + "  ".join(f"{k}={v:.1e}" for k, v in errs.items())
                    )
            except AssertionError as exc:
                print(f"FAIL trial={trial} strategy={strategy}: {exc}")
            except Exception as exc:
                print(f"ERROR trial={trial} strategy={strategy}: {type(exc).__name__}: {exc}")

    print(f"\n{'PASSED' if passed == total else 'FAILED'} {passed}/{total} "
          f"({n_trials} splines × {len(STRATEGIES)} strategies)")
    if max_errs:
        print("Max errors across all trials:")
        for name, err in sorted(max_errs.items()):
            print(f"  {name}: {err:.3e}")
    return passed == total


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    ok = run_tests(verbose=verbose)
    sys.exit(0 if ok else 1)
