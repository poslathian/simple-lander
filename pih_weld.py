"""C2-continuous B-spline welding for receding-horizon trajectory splicing.

Pure numpy/scipy — no pydrake, no env dependencies.
Works with arbitrary N control points (degree-3 clamped B-splines).

Three strategies (WRITEUP.md §Welding Solution / §Overlap Welding):
  "point"     — 3 constrained CPs (P0,P1,P2), N-3 free. Exact C2 via closed-form.
  "overlap_1" — 4 constrained CPs, N-4 free. Adds P3=old(weld+Δ) so span 1 ends
                on the old trajectory; structural C2 at interior knot.
  "overlap_2" — 5 constrained CPs, N-5 free. Adds P3, P4 to cover 2 spans.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def _make_uniform_clamped_knots(n_cps: int, delta: float) -> np.ndarray:
    """Clamped cubic B-spline knot vector: n_cps control points, uniform spacing delta.

    Layout: [0,0,0,0, Δ,2Δ,...,(n_cps-4)Δ, T_max,T_max,T_max,T_max]
    where T_max = (n_cps-3)*delta.  Total knot count = n_cps+4.
    """
    n_interior = n_cps - 4
    t_max = (n_cps - 3) * delta
    interior = np.arange(1, n_interior + 1, dtype=float) * delta
    return np.concatenate([[0.0] * 4, interior, [t_max] * 4])


def _eval_basis_at(knots: np.ndarray, n_cps: int, t: float, indices) -> np.ndarray:
    """Evaluate B-spline basis functions N_i(t) for each i in indices.

    Uses the unit-coefficient trick: the value of basis function N_i at t
    equals BSpline(knots, e_i, 3)(t) where e_i is the i-th standard basis vector.
    """
    row = np.empty(len(indices))
    c = np.zeros(n_cps)
    for k, i in enumerate(indices):
        c[i] = 1.0
        row[k] = float(BSpline(knots, c, 3)(t))
        c[i] = 0.0
    return row


def weld_c2(
    old_cps: np.ndarray,         # (N, D) control points of executing spline
    old_knots: np.ndarray,       # (N+4,) knot vector (degree-3 clamped)
    new_free_cps: np.ndarray,    # (N-k, D) free CPs from oracle; k=3/4/5 per strategy
    new_knot_spacing: float,     # Δ for the new uniform spline
    weld_param: float,           # parameter on old spline where weld occurs
    strategy: str = "point",     # "point", "overlap_1", "overlap_2"
) -> tuple[np.ndarray, np.ndarray]:
    """Weld new_free_cps onto old spline at weld_param with C2 continuity.

    Returns (new_cps, new_knots):
      new_cps  — (N, D) complete welded control points
      new_knots — (N+4,) uniform clamped knot vector for the new spline

    Verification (WRITEUP.md §Verification):
      |S_new(0)    - S_old(weld_param)|   < 1e-10  ← C0
      |S_new'(0)   - S_old'(weld_param)|  < 1e-10  ← C1
      |S_new''(0)  - S_old''(weld_param)| < 1e-10  ← C2
    """
    N, D = old_cps.shape
    delta = new_knot_spacing

    _n_constrained = {"point": 3, "overlap_1": 4, "overlap_2": 5}
    if strategy not in _n_constrained:
        raise ValueError(f"Unknown strategy {strategy!r}; must be one of {sorted(_n_constrained)}")
    k = _n_constrained[strategy]
    expected_free = N - k
    if new_free_cps.shape != (expected_free, D):
        raise ValueError(
            f"strategy={strategy!r} needs {expected_free} free CPs of dim {D}; "
            f"got shape {new_free_cps.shape}"
        )

    # Evaluate old spline at weld_param using scipy — recommended by WRITEUP.md
    spl = BSpline(old_knots, old_cps, 3)
    pos = spl(weld_param)                      # (D,)
    vel = spl.derivative(1)(weld_param)        # (D,)
    acc = spl.derivative(2)(weld_param)        # (D,)

    # ── C2-constrained first 3 CPs — WRITEUP.md §Solving for Constrained CPs ──
    # P0 = pos
    # P1 = pos + vel·Δ/3
    # P2 = pos + vel·Δ + acc·Δ²/3
    P0 = pos
    P1 = pos + vel * (delta / 3.0)
    P2 = pos + vel * delta + acc * (delta * delta / 3.0)
    constrained = [P0, P1, P2]

    if strategy == "overlap_1":
        # P3 pins the first span's endpoint to the old trajectory at weld_param+Δ.
        # B-spline property: S_new(Δ) = P3 (last Bézier CP of clamped first span).
        P3 = spl(weld_param + delta)
        constrained.append(P3)

    elif strategy == "overlap_2":
        # P3 as for overlap_1
        P3 = spl(weld_param + delta)
        constrained.append(P3)

        # P4: solve S_new(2Δ) = S_old(weld_param + 2Δ).
        # S_new(2Δ) = Σ_{i=0}^{4} N_i(2Δ)·P_i ; P_0..P_3 known, solve for P_4.
        new_knots_tmp = _make_uniform_clamped_knots(N, delta)
        basis_at_2d = _eval_basis_at(new_knots_tmp, N, 2.0 * delta, range(5))
        n4_val = basis_at_2d[4]
        if abs(n4_val) < 1e-14:
            raise RuntimeError(
                f"N_4(2Δ) ≈ 0 (got {n4_val:.2e}); cannot solve for P4 in overlap_2"
            )
        target = spl(weld_param + 2.0 * delta)
        known_contrib = (
            basis_at_2d[0] * P0
            + basis_at_2d[1] * P1
            + basis_at_2d[2] * P2
            + basis_at_2d[3] * P3
        )
        P4 = (target - known_contrib) / n4_val
        constrained.append(P4)

    new_cps = np.vstack([np.array(constrained), new_free_cps])
    new_knots = _make_uniform_clamped_knots(N, delta)
    return new_cps, new_knots
