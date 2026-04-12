"""
C2-continuous B-spline welding.

Extends a clamped cubic B-spline (10 control points) with a new 10-CP spline
such that position, velocity, and acceleration are continuous at the junction.

Two strategies:

**Point weld** (weld_bspline_c2):
    First 3 CPs determined by C2 constraint at the junction.
    Model predicts 7 free CPs. C2 at the junction via closed-form.

**Overlap weld** (overlap_weld_bspline_c2):
    First k*4 (1-span) or 5 (2-span) CPs match the old spline's tail.
    C2 at the switchover is structural (B-spline interior knot continuity).
    Absorbs k knot spans of inference delay.

Closed-form (point weld, uniform Δ):
    P_0 = pos
    P_1 = pos + vel * Δ/3
    P_2 = pos + vel * Δ + acc * Δ²/3
    P_3 ... P_9 = free_cps[0] ... free_cps[6]

Bézier copy (1-span overlap):
    P_0..P_3 = old_cps[6..9]
    P_4 ... P_9 = free_cps[0] ... free_cps[5]
"""

from typing import Literal, overload
import numpy as np
from numpy.typing import NDArray

def weld_bspline_c2(
    old_cps: NDArray[np.floating],
    old_knots: NDArray[np.floating],
    free_cps: NDArray[np.floating],
    new_knot_spacing: float,
    weld_param: float | None = None,
) -> NDArray[np.floating]:
    """Weld a new 10-CP cubic B-spline onto an existing one with C2 continuity.

    Evaluates the source spline at the weld point to extract (pos, vel, acc),
    computes the 3 constrained control points, and concatenates with the 7
    free control points to produce the full 10-CP result.

    Args:
        old_cps: Source spline control points, shape (10, D) where D is the
            spatial dimension (e.g. 2 for xy, 3 for xyz, 6 for full state).
        old_knots: Source spline knot vector, shape (14,). Must be a valid
            clamped cubic B-spline knot vector (multiplicity 4 at each end).
        free_cps: Model-predicted free control points for the new spline,
            shape (7, D). These become P_3 through P_9 of the output.
        new_knot_spacing: Uniform knot spacing Δ for the new spline's knot
            vector. The new knot vector is constructed as:
            [0,0,0,0, Δ, 2Δ, 3Δ, 4Δ, 5Δ, 6Δ, 7Δ, 7Δ, 7Δ, 7Δ]
        weld_param: Parameter value on the old spline at which to weld.
            If None (default), welds at the right endpoint (old_knots[-1]).

    Returns:
        Welded control points, shape (10, D). The first 3 rows are the
        constrained CPs; the last 7 are free_cps unchanged.

    Raises:
        ValueError: If old_cps is not (10, D), free_cps is not (7, D),
            old_knots is not (14,), or dimensions don't match.
    """
    ...


def extract_weld_state(
    cps: NDArray[np.floating],
    knots: NDArray[np.floating],
    t: float | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Extract position, velocity, and acceleration from a B-spline at a point.

    Uses scipy.interpolate.BSpline for robust evaluation that handles
    non-uniform knots and arbitrary parameter values.

    Args:
        cps: Control points, shape (N, D).
        knots: Knot vector, shape (N+4,) for cubic B-spline.
        t: Parameter value at which to evaluate. If None, evaluates at the
            right endpoint (knots[-1]).

    Returns:
        Tuple of (pos, vel, acc), each shape (D,):
            pos: S(t) — position
            vel: S'(t) — first derivative
            acc: S''(t) — second derivative
    """
    ...


def make_clamped_knots(
    n_cps: int,
    spacing: float,
    degree: int = 3,
) -> NDArray[np.floating]:
    """Construct a clamped uniform knot vector.

    Args:
        n_cps: Number of control points (e.g. 10).
        spacing: Uniform interior knot spacing Δ.
        degree: B-spline degree (default 3 for cubic).

    Returns:
        Knot vector, shape (n_cps + degree + 1,). For n_cps=10, degree=3,
        this is shape (14,):
        [0, 0, 0, 0, Δ, 2Δ, ..., (n_cps-degree-1)*Δ, ..., repeated]
    """
    ...


def constrained_cps(
    pos: NDArray[np.floating],
    vel: NDArray[np.floating],
    acc: NDArray[np.floating],
    knot_spacing: float,
) -> NDArray[np.floating]:
    """Compute the 3 constrained control points for a C2 weld.

    These are the closed-form solutions:
        P_0 = pos
        P_1 = pos + vel * Δ/3
        P_2 = pos + vel * Δ + acc * Δ²/3

    This is the pure-math kernel — no spline evaluation, no concatenation.
    Useful when you already have (pos, vel, acc) from another source
    (e.g., simulator state rather than a previous spline).

    Args:
        pos: Position at weld point, shape (D,).
        vel: Velocity at weld point, shape (D,).
        acc: Acceleration at weld point, shape (D,).
        knot_spacing: Uniform knot spacing Δ for the new spline.

    Returns:
        Constrained control points, shape (3, D).
    """
    ...


def constrained_cps_nonuniform(
    pos: NDArray[np.floating],
    vel: NDArray[np.floating],
    acc: NDArray[np.floating],
    h1: float,
    h2: float,
) -> NDArray[np.floating]:
    """Compute constrained CPs for non-uniform knot spacing.

    Generalized formulas:
        P_0 = pos
        P_1 = pos + vel * h1 / 3
        P_2 = P_1 + h2 * (acc * h1 / 6 + vel / 3)

    where h1 = t_4 (first interior knot) and h2 = t_5 (second interior knot)
    of the new spline's knot vector.

    Collapses to constrained_cps() when h1 = Δ and h2 = 2Δ.

    Args:
        pos: Position at weld point, shape (D,).
        vel: Velocity at weld point, shape (D,).
        acc: Acceleration at weld point, shape (D,).
        h1: First interior knot value (t_4) of the new spline.
        h2: Second interior knot value (t_5) of the new spline.

    Returns:
        Constrained control points, shape (3, D).
    """
    ...


def overlap_weld_bspline_c2(
    old_cps: NDArray[np.floating],
    old_knots: NDArray[np.floating],
    free_cps: NDArray[np.floating],
    new_knot_spacing: float,
    overlap_spans: Literal[1, 2] = 1,
) -> NDArray[np.floating]:
    """Weld with overlap: new spline replays old spline's tail, then diverges.

    The first `overlap_spans` spans of the new spline exactly reproduce the
    last `overlap_spans` spans of the old spline. C2 continuity at the
    switchover (end of overlap) is guaranteed by B-spline interior knot
    continuity — no derivative matching needed.

    Use this when inference takes ~overlap_spans * Δ seconds: the overlap
    absorbs the computation delay, and free CPs take effect at the actual
    switchover time.

    Args:
        old_cps: Source spline control points, shape (10, D).
        old_knots: Source spline knot vector, shape (14,). Clamped cubic.
        free_cps: Model-predicted free control points.
            - 1-span overlap: shape (6, D) — becomes P_4 through P_9.
            - 2-span overlap: shape (5, D) — becomes P_5 through P_9.
        new_knot_spacing: Uniform knot spacing Δ for the new spline.
            Must equal the old spline's last span width for 1-span overlap.
        overlap_spans: Number of old-spline tail spans to replay.
            1: copy last 4 old CPs (Bézier identity, exact).
            2: solve 5×5 collocation system (exact to machine precision).

    Returns:
        Welded control points, shape (10, D).
        - 1-span: [P_6^old, P_7^old, P_8^old, P_9^old, free_0..free_5]
        - 2-span: [P_0..P_4 from collocation, free_0..free_4]

    Note:
        The new spline's valid region starts at t=0, but the overlap region
        [0, overlap_spans * Δ] reproduces the old spline. Evaluate from
        t = overlap_spans * Δ onward for the model's new trajectory.
    """
    ...


def overlap_constrained_cps(
    old_cps: NDArray[np.floating],
    old_knots: NDArray[np.floating],
    new_knot_spacing: float,
    overlap_spans: Literal[1, 2] = 1,
) -> NDArray[np.floating]:
    """Compute just the constrained CPs for an overlap weld.

    Pure-math kernel — returns the CPs without concatenating free CPs.

    For 1-span overlap: returns old_cps[-4:] (Bézier copy, trivial).
    For 2-span overlap: solves 5-point collocation against old spline.

    Args:
        old_cps: Source spline control points, shape (10, D).
        old_knots: Source spline knot vector, shape (14,).
        new_knot_spacing: Uniform knot spacing Δ.
        overlap_spans: 1 or 2.

    Returns:
        Constrained CPs, shape (4, D) for 1-span or (5, D) for 2-span.
    """
    ...


def verify_c2_weld(
    old_cps: NDArray[np.floating],
    old_knots: NDArray[np.floating],
    new_cps: NDArray[np.floating],
    new_knots: NDArray[np.floating],
    weld_param: float | None = None,
    atol: float = 1e-10,
) -> dict[str, bool | float]:
    """Verify C2 continuity at the weld junction.

    Evaluates both splines at the junction and checks that position,
    velocity, and acceleration match within tolerance.

    Args:
        old_cps: Source spline control points, shape (10, D).
        old_knots: Source spline knot vector, shape (14,).
        new_cps: Welded spline control points, shape (10, D).
        new_knots: New spline knot vector, shape (14,).
        weld_param: Parameter on old spline where weld occurs.
            Default: right endpoint.
        atol: Absolute tolerance for continuity checks.

    Returns:
        Dict with keys:
            "c0_ok": bool — position match
            "c1_ok": bool — velocity match
            "c2_ok": bool — acceleration match
            "c0_err": float — position error norm
            "c1_err": float — velocity error norm
            "c2_err": float — acceleration error norm
    """
    ...


def verify_overlap_weld(
    old_cps: NDArray[np.floating],
    old_knots: NDArray[np.floating],
    new_cps: NDArray[np.floating],
    new_knots: NDArray[np.floating],
    overlap_spans: int = 1,
    n_samples: int = 50,
    atol: float = 1e-10,
) -> dict[str, bool | float]:
    """Verify an overlap weld by checking agreement in the overlap region.

    Samples both splines at n_samples points in the overlap region and
    checks that position, velocity, and acceleration agree. Also checks
    C2 at the switchover boundary.

    Args:
        old_cps: Source spline control points, shape (10, D).
        old_knots: Source spline knot vector, shape (14,).
        new_cps: Welded spline control points, shape (10, D).
        new_knots: New spline knot vector, shape (14,).
        overlap_spans: Number of overlap spans (1 or 2).
        n_samples: Points to sample in the overlap region.
        atol: Absolute tolerance.

    Returns:
        Dict with keys:
            "overlap_max_pos_err": float — max position error in overlap
            "overlap_max_vel_err": float — max velocity error in overlap
            "overlap_max_acc_err": float — max acceleration error in overlap
            "c2_at_switchover": bool — C2 holds at the overlap boundary
            "switchover_c0_err": float
            "switchover_c1_err": float
            "switchover_c2_err": float
    """
    ...
