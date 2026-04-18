"""PIH solver: KTO trajectory planner and tracking controller for PackageInHoleEnv."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.interpolate import BSpline as _ScipyBSpline
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import LinearEqualityConstraint, Solve

from kto_lander import Plan, LanderParams, TrackerGains, control
from pih_env import (
    PIHConfig,
    PIH_START_X, PIH_PICKUP_X, PIH_PAD_Y, PIH_LEG_OFFSET,
    PIH_MOUNTAIN_X, PIH_MOUNTAIN_H,
    DT,
)
from pih_weld import weld_c2, _make_uniform_clamped_knots

# Clearance above the mountain peak (m).
_MOUNTAIN_CLEARANCE_M = 1.5

# Default normalised times for the 7 PIH waypoints (approach_land is a read-back).
_DEFAULT_S: tuple[float, ...] = (0.00, 0.20, 0.40, 0.52, 0.62, 0.77, 0.88, 1.00)

# Proximity thresholds (metres) per waypoint for the controller's index advance.
_WP_THRESHOLDS: dict[str, float] = {
    "mountain_out":  2.5,
    "approach":      2.0,
    "contact":       1.5,
    "extraction":    0.8,
    "mountain_ret":  2.5,
    "approach_land": 2.0,
    "landing":       1.5,
}

# Canonical ordering used by _tracked_waypoints (approach_land excluded — it
# is a read-back annotation, not a physical navigation target).
_WP_ORDER = ["mountain_out", "approach", "contact", "extraction",
             "mountain_ret", "landing"]


# ---------------------------------------------------------------------------
# Scipy-backed trajectory wrapper — pydrake-compatible interface
# ---------------------------------------------------------------------------

class _ScipyBasisWrapper:
    """Thin wrapper so _ScipyTrajWrapper.basis().knots() works."""
    def __init__(self, knots: np.ndarray) -> None:
        self._knots = knots

    def knots(self):
        return self._knots.tolist()


class _ScipyTrajWrapper:
    """Scipy BSpline wrapped with pydrake BsplineTrajectory interface.

    Returned by _numpy_to_plan() so welded plans can be used in Plan objects
    without requiring a round-trip through pydrake's BsplineTrajectory ctor.
    """
    def __init__(self, spl: _ScipyBSpline, T: float) -> None:
        self._spl = spl
        self._T = T

    def value(self, t: float) -> np.ndarray:
        """Returns (D, 1) column vector, matching pydrake convention."""
        v = self._spl(float(np.clip(t, 0.0, self._T)))
        return np.asarray(v, dtype=float).reshape(-1, 1)

    def MakeDerivative(self, order: int) -> "_ScipyTrajWrapper":
        return _ScipyTrajWrapper(self._spl.derivative(order), self._T)

    def control_points(self) -> list[np.ndarray]:
        c = self._spl.c          # (N, D) for multi-dim spline
        return [c[i:i+1, :].T for i in range(len(c))]   # list of (D, 1)

    def basis(self) -> _ScipyBasisWrapper:
        return _ScipyBasisWrapper(self._spl.t)

    def end_time(self) -> float:
        return self._T

    def start_time(self) -> float:
        return 0.0


def _plan_to_numpy(plan: Plan) -> tuple[np.ndarray, np.ndarray]:
    """Extract (N, D) control points and (N+4,) knot vector from a Plan."""
    cps_raw = plan.traj.control_points()          # list of (D, 1) arrays
    cps = np.array([np.asarray(cp).flatten() for cp in cps_raw])  # (N, D)
    knots = np.asarray(plan.traj.basis().knots(), dtype=float)     # (N+4,)
    return cps, knots


def _numpy_to_plan(cps: np.ndarray, knots: np.ndarray) -> Plan:
    """Build a Plan from (N, D) control points and (N+4,) knot vector."""
    T = float(knots[-1])
    spl = _ScipyBSpline(knots, cps, 3)
    traj = _ScipyTrajWrapper(spl, T)
    return Plan(traj, T)


# ---------------------------------------------------------------------------
# PIHWaypoints
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PIHWaypoints:
    """Named milestones along the PIH plan, each as (x_world, y_world, s_normalised).

    s ∈ [0, 1] is the normalised KTO time; multiply by plan.T to get wall-clock
    seconds.  All coordinates are absolute world metres (not pad-centred).

    Fields not active in a partial (oracle) plan are set to (nan, nan, nan).
    """
    start:         tuple[float, float, float]
    mountain_out:  tuple[float, float, float]
    approach:      tuple[float, float, float]
    contact:       tuple[float, float, float]
    extraction:    tuple[float, float, float]
    mountain_ret:  tuple[float, float, float]
    approach_land: tuple[float, float, float]
    landing:       tuple[float, float, float]

    def as_list(self) -> list[tuple[float, float, float]]:
        return [getattr(self, f.name) for f in dataclasses.fields(self)]

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


# ---------------------------------------------------------------------------
# plan_pih_with_kto
# ---------------------------------------------------------------------------

def _compute_wpt_positions(x0: float, y0: float, cfg: PIHConfig) -> dict[str, tuple[float, float]]:
    """Compute the world-frame (x, y) position for every named PIH waypoint."""
    mountain_y_out = PIH_PAD_Y + PIH_MOUNTAIN_H + _MOUNTAIN_CLEARANCE_M + 0.5
    mountain_y_ret = mountain_y_out + cfg.package_height_assumed
    approach_y     = cfg.assumed_contact_lander_y + 2.5
    landing_y      = PIH_PAD_Y + cfg.package_height_assumed + PIH_LEG_OFFSET
    return {
        "start":        (x0,            y0),
        "mountain_out": (PIH_MOUNTAIN_X, mountain_y_out),
        "approach":     (PIH_PICKUP_X,   approach_y),
        "contact":      (PIH_PICKUP_X,   cfg.assumed_contact_lander_y),
        "extraction":   (PIH_PICKUP_X,   cfg.extraction_lander_y),
        "mountain_ret": (PIH_MOUNTAIN_X, mountain_y_ret),
        "landing":      (PIH_START_X,    landing_y),
    }


def plan_pih_with_kto(
    x0: float,
    y0: float,
    vx0: float,
    vy0: float,
    cfg: PIHConfig,
    params: LanderParams | None = None,
    *,
    remaining_waypoints: list[str] | None = None,
    waypoints_s: tuple[float, ...] | None = None,
    num_control_points: int = 20,
    spline_order: int = 4,
    T_min: float | None = None,
    T_max: float = 30.0,
    vy_contact: float = -0.1,
    vy_extraction: float = 0.5,
    vy_landing: float = -0.3,
    verbose: bool = True,
) -> tuple[Plan, float, PIHWaypoints]:
    """Solve a PIH trajectory via pydrake KTO.

    When remaining_waypoints is None (default): plans through all 7 waypoints
    using _DEFAULT_S normalised times — identical to original Phase 1 behaviour.

    When remaining_waypoints is a list of waypoint names: plans only through
    those waypoints starting from (x0, y0).  Used by the oracle replan.

    Returns (Plan, duration_T_seconds, PIHWaypoints).
    Raises RuntimeError on solver failure.
    """
    wpt_xy = _compute_wpt_positions(x0, y0, cfg)
    _NaN3 = (float("nan"), float("nan"), float("nan"))

    # ── Build active waypoint list and normalised times ───────────────────
    if remaining_waypoints is None:
        # Original full-plan path
        if T_min is None:
            T_min = 10.0
        if waypoints_s is None:
            waypoints_s = _DEFAULT_S
        if len(waypoints_s) != 8:
            raise ValueError(f"waypoints_s must have 8 entries for full plan, got {len(waypoints_s)}")
        s0, s_mo, s_ap, s_ct, s_ex, s_mr, s_al, s1 = waypoints_s

        active_wpts: dict[str, tuple[float, float, float]] = {
            "start":        (wpt_xy["start"][0],        wpt_xy["start"][1],        s0),
            "mountain_out": (wpt_xy["mountain_out"][0], wpt_xy["mountain_out"][1], s_mo),
            "approach":     (wpt_xy["approach"][0],     wpt_xy["approach"][1],     s_ap),
            "contact":      (wpt_xy["contact"][0],      wpt_xy["contact"][1],      s_ct),
            "extraction":   (wpt_xy["extraction"][0],   wpt_xy["extraction"][1],   s_ex),
            "mountain_ret": (wpt_xy["mountain_ret"][0], wpt_xy["mountain_ret"][1], s_mr),
            "landing":      (wpt_xy["landing"][0],      wpt_xy["landing"][1],      s1),
        }
        s_al_read = s_al

        vel_constraints = [
            (vx0, vy0,          s0),
            (0.0, vy_contact,   s_ct),
            (0.0, vy_extraction, s_ex),
            (0.0, vy_landing,   s1),
        ]

    else:
        # Partial plan for oracle replan
        if T_min is None:
            T_min = 10.0
        valid = set(wpt_xy.keys()) - {"start"}
        unknown = [n for n in remaining_waypoints if n not in valid]
        if unknown:
            raise ValueError(f"Unknown waypoints: {unknown}")

        ordered = [n for n in _WP_ORDER if n in remaining_waypoints]
        n_active = 1 + len(ordered)  # start + ordered remaining

        if waypoints_s is not None:
            if len(waypoints_s) != n_active:
                raise ValueError(
                    f"waypoints_s needs {n_active} entries for this partial plan, "
                    f"got {len(waypoints_s)}"
                )
            s_vals = list(waypoints_s)
        else:
            # Evenly-spaced times: gives each segment equal share of T_min,
            # avoiding infeasibility when a waypoint is geometrically close but
            # dynamically hard to reach (e.g. extraction requires upward velocity reversal).
            s_vals = list(np.linspace(0.0, 1.0, n_active))

        active_wpts = {"start": (x0, y0, s_vals[0])}
        for i, name in enumerate(ordered):
            wx, wy = wpt_xy[name]
            active_wpts[name] = (wx, wy, s_vals[i + 1])

        s0 = s_vals[0]
        s1 = s_vals[-1]
        s_al_read = None

        vel_constraints = [(vx0, vy0, s0)]
        if "contact"    in active_wpts: vel_constraints.append((0.0, vy_contact,    active_wpts["contact"][2]))
        if "extraction" in active_wpts: vel_constraints.append((0.0, vy_extraction, active_wpts["extraction"][2]))
        vel_constraints.append((0.0, vy_landing, s1))

    # ── Build KTO problem ────────────────────────────────────────────────
    ax_max, ay_min, ay_max = 1.5, -1.0, 5.0
    vx_max, vy_min, vy_max = 7.0, -7.0, 5.0

    kto = KinematicTrajectoryOptimization(
        num_positions=2,
        num_control_points=num_control_points,
        spline_order=spline_order,
    )
    kto.AddDurationConstraint(T_min, T_max)
    kto.AddDurationCost(1.0)
    # Full plans use 0.02 path-length weight (original value).
    # Partial (oracle) plans use 0.001 — enough to prevent lateral detours through
    # the extraction zone without triggering SNOPT's kSolverSpecificError.
    path_len_weight = 0.02 if remaining_waypoints is None else 0.001
    kto.AddPathLengthCost(path_len_weight)
    kto.AddVelocityBounds(np.array([-vx_max, vy_min]), np.array([vx_max, vy_max]))
    kto.AddAccelerationBounds(np.array([-ax_max, ay_min]), np.array([ax_max, ay_max]))

    for wx, wy, ws in active_wpts.values():
        kto.AddPathPositionConstraint(np.array([wx, wy]), np.array([wx, wy]), ws)

    zero2 = np.zeros(2)
    kto.AddPathAccelerationConstraint(zero2, zero2, s0)
    kto.AddPathAccelerationConstraint(zero2, zero2, s1)

    def _vel_eq(vx_req: float, vy_req: float) -> LinearEqualityConstraint:
        A = np.zeros((2, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        return LinearEqualityConstraint(A, np.array([vx_req, vy_req]))

    for vx_req, vy_req, s_req in vel_constraints:
        kto.AddVelocityConstraintAtNormalizedTime(_vel_eq(vx_req, vy_req), s_req)

    result = Solve(kto.prog())
    if not result.is_success():
        if verbose:
            print(f"[pih_kto] FAILED: {result.get_solver_id().name()} "
                  f"{result.get_solution_result()}")
        raise RuntimeError(f"PIH KTO solve failed: {result.get_solution_result()}")

    traj = kto.ReconstructTrajectory(result)
    T = float(traj.end_time() - traj.start_time())

    # Read back approach_land from the trajectory if available
    if s_al_read is not None:
        al_q = traj.value(s_al_read * T).flatten()
        al_wpt = (float(al_q[0]), float(al_q[1]), s_al_read)
    else:
        al_wpt = _NaN3

    wpts = PIHWaypoints(
        start         = active_wpts.get("start",        _NaN3),
        mountain_out  = active_wpts.get("mountain_out", _NaN3),
        approach      = active_wpts.get("approach",     _NaN3),
        contact       = active_wpts.get("contact",      _NaN3),
        extraction    = active_wpts.get("extraction",   _NaN3),
        mountain_ret  = active_wpts.get("mountain_ret", _NaN3),
        approach_land = al_wpt,
        landing       = active_wpts.get("landing",      _NaN3),
    )

    if verbose:
        label = "oracle" if remaining_waypoints is not None else "kto"
        print(f"[pih_{label}] solved: T={T:.2f}s  #cp={num_control_points}  "
              f"solver={result.get_solver_id().name()}")
        for f in dataclasses.fields(wpts):
            wx, wy, ws = getattr(wpts, f.name)
            if not math.isnan(wx):
                print(f"  {f.name:<14s} ({wx:6.2f}, {wy:5.2f})  s={ws:.2f}  t={ws * T:5.1f}s")

    return Plan(traj, T), T, wpts


# ---------------------------------------------------------------------------
# Oracle KTO
# ---------------------------------------------------------------------------

def oracle_plan_pih_with_kto(
    current_state: np.ndarray,        # (6,) [x, y, vx, vy, theta, omega] — theta/omega unused
    cfg: PIHConfig,
    remaining_waypoints: list[str],
    *,
    verbose: bool = False,
    **kwargs,
) -> tuple[Plan, float, PIHWaypoints]:
    """Replan from current_state using TRUE parameters (oracle config).

    Thin wrapper around plan_pih_with_kto with:
      - oracle_cfg = cfg.with_true_as_assumed()
      - start position/velocity from current_state
      - only plans through remaining_waypoints

    Returns (Plan, T, PIHWaypoints) — same type as plan_pih_with_kto.
    """
    oracle_cfg = cfg.with_true_as_assumed()
    x0, y0, vx0, vy0 = (float(current_state[0]), float(current_state[1]),
                         float(current_state[2]), float(current_state[3]))
    return plan_pih_with_kto(
        x0, y0, vx0, vy0,
        oracle_cfg,
        remaining_waypoints=remaining_waypoints,
        verbose=verbose,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Waypoint re-solve (Phase 2.5)
# ---------------------------------------------------------------------------

def waypoint_replan_pih_with_kto(
    user_waypoint: np.ndarray,
    previous_waypoint_name: str,
    previous_waypoint_state: np.ndarray,
    cfg: PIHConfig,
    remaining_waypoints: list[str],
    params: LanderParams | None = None,
    *,
    waypoint_tolerance: float = 0.5,
    T_min: float = 5.0,
    T_max: float = 30.0,
    verbose: bool = False,
) -> dict:
    """Replan from previous_waypoint through user_waypoint using ASSUMED parameters.

    The user waypoint is inserted as an intermediate soft position constraint
    (bounding box ±waypoint_tolerance metres) between previous_waypoint and the
    first remaining named waypoint.  All remaining named waypoints are added as
    hard position constraints.

    Parameters
    ----------
    user_waypoint : (2,) array  — (x, y) in world metres, from operator click
    previous_waypoint_name : str  — e.g. "contact"
    previous_waypoint_state : (6,) array  — [x, y, vx, vy, theta, omega]
    cfg : PIHConfig  — ASSUMED parameters (NOT oracle/true)
    remaining_waypoints : list of named waypoints after the user waypoint
    waypoint_tolerance : soft-constraint half-width in metres (default 0.5)

    Returns
    -------
    dict with keys:
      feasible : bool
      plan     : Plan (if feasible)
      T        : float — plan duration seconds (if feasible)
      waypoints: PIHWaypoints (if feasible)
      cps      : np.ndarray (N, 2)  — control points (if feasible)
      knots    : np.ndarray (N+4,)  — knot vector (if feasible)
    """
    x0 = float(previous_waypoint_state[0])
    y0 = float(previous_waypoint_state[1])
    vx0 = float(previous_waypoint_state[2])
    vy0 = float(previous_waypoint_state[3])

    uw_x = float(user_waypoint[0])
    uw_y = float(user_waypoint[1])

    # Compute remaining named waypoint positions using ASSUMED config
    wpt_xy = _compute_wpt_positions(x0, y0, cfg)

    # Build ordered remaining list (only those actually in _WP_ORDER)
    ordered_remaining = [n for n in _WP_ORDER if n in remaining_waypoints]

    # Estimate normalised time for user waypoint by distance proportion
    prev_pos = np.array([x0, y0])
    uw_pos   = np.array([uw_x, uw_y])

    seg_lens = [np.linalg.norm(uw_pos - prev_pos)]
    cur = uw_pos
    for name in ordered_remaining:
        nxt = np.array(wpt_xy.get(name, (cur[0], cur[1])))
        seg_lens.append(float(np.linalg.norm(nxt - cur)))
        cur = nxt
    total = sum(seg_lens) or 1.0
    s_user = seg_lens[0] / total

    # Normalised times for named waypoints (evenly distributed in [s_user, 1])
    n_after = len(ordered_remaining)
    if n_after > 0:
        s_after = list(np.linspace(s_user + (1.0 - s_user) / (n_after + 1),
                                   1.0, n_after))
    else:
        s_after = []

    # ── Build KTO problem ────────────────────────────────────────────────
    num_control_points = 20
    spline_order = 4
    kto = KinematicTrajectoryOptimization(
        num_positions=2,
        num_control_points=num_control_points,
        spline_order=spline_order,
    )
    kto.AddDurationConstraint(T_min, T_max)
    kto.AddDurationCost(1.0)
    kto.AddPathLengthCost(0.001)
    kto.AddVelocityBounds(np.array([-7.0, -7.0]), np.array([7.0, 5.0]))
    kto.AddAccelerationBounds(np.array([-1.5, -1.0]), np.array([1.5, 5.0]))

    # Start position (hard)
    kto.AddPathPositionConstraint(np.array([x0, y0]), np.array([x0, y0]), 0.0)

    # User waypoint (soft — tolerance box)
    tol = waypoint_tolerance
    kto.AddPathPositionConstraint(
        np.array([uw_x - tol, uw_y - tol]),
        np.array([uw_x + tol, uw_y + tol]),
        s_user,
    )

    # Named remaining waypoints (hard)
    for name, s_req in zip(ordered_remaining, s_after):
        wx, wy = wpt_xy[name]
        kto.AddPathPositionConstraint(np.array([wx, wy]), np.array([wx, wy]), s_req)

    # Zero acceleration at endpoints
    zero2 = np.zeros(2)
    kto.AddPathAccelerationConstraint(zero2, zero2, 0.0)
    kto.AddPathAccelerationConstraint(zero2, zero2, 1.0)

    # Velocity constraints
    def _vel_eq(vx_req: float, vy_req: float) -> LinearEqualityConstraint:
        A = np.zeros((2, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        return LinearEqualityConstraint(A, np.array([vx_req, vy_req]))

    kto.AddVelocityConstraintAtNormalizedTime(_vel_eq(vx0, vy0), 0.0)
    if "extraction" in ordered_remaining:
        s_ex = s_after[ordered_remaining.index("extraction")]
        kto.AddVelocityConstraintAtNormalizedTime(_vel_eq(0.0, 0.5), s_ex)
    kto.AddVelocityConstraintAtNormalizedTime(_vel_eq(0.0, -0.3), 1.0)

    result = Solve(kto.prog())
    if not result.is_success():
        if verbose:
            print(f"[pih_wp_replan] FAILED: {result.get_solver_id().name()} "
                  f"{result.get_solution_result()}")
        return {"feasible": False}

    traj = kto.ReconstructTrajectory(result)
    T = float(traj.end_time() - traj.start_time())

    _NaN3 = (float("nan"), float("nan"), float("nan"))
    active: dict[str, tuple] = {}
    active[previous_waypoint_name] = (x0, y0, 0.0)
    for name, s_req in zip(ordered_remaining, s_after):
        wx, wy = wpt_xy[name]
        active[name] = (wx, wy, s_req)

    wpts = PIHWaypoints(
        start         = (x0, y0, 0.0),
        mountain_out  = active.get("mountain_out", _NaN3),
        approach      = active.get("approach",     _NaN3),
        contact       = active.get("contact",      _NaN3),
        extraction    = active.get("extraction",   _NaN3),
        mountain_ret  = active.get("mountain_ret", _NaN3),
        approach_land = _NaN3,
        landing       = active.get("landing",      _NaN3),
    )

    plan_obj = Plan(traj, T)
    cps, knots = _plan_to_numpy(plan_obj)

    if verbose:
        print(f"[pih_wp_replan] solved: T={T:.2f}s  solver={result.get_solver_id().name()}")
        print(f"  prev={previous_waypoint_name}  user_wp=({uw_x:.2f},{uw_y:.2f})  "
              f"s_user={s_user:.2f}  remaining={ordered_remaining}")

    return {
        "feasible":  True,
        "plan":      plan_obj,
        "T":         T,
        "waypoints": wpts,
        "cps":       cps,
        "knots":     knots,
    }


# ---------------------------------------------------------------------------
# _HoverPlan — fake terminal plan
# ---------------------------------------------------------------------------

class _HoverPlan:
    """Fake plan that holds the terminal landing position with zero velocity/accel."""

    def __init__(self, landing_x: float, landing_y: float) -> None:
        self.T = 0.0
        self._ref = (landing_x, landing_y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __call__(self, t: float):  # noqa: ANN201
        return self._ref


# ---------------------------------------------------------------------------
# PackageInHoleKTOController
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _TrackedWaypoint:
    name:      str
    x:         float
    y:         float
    threshold: float
    vy_min:    float = -float("inf")  # require vy >= vy_min to advance past this waypoint


# extraction is at the same x as contact but higher y — the lander passes through
# its vicinity while DESCENDING to contact.  Require ascending velocity so the
# index only advances past extraction on the genuine ascent phase.
_WP_VY_MIN: dict[str, float] = {"extraction": 0.0}


def _waypoints_from_pih(wpts: PIHWaypoints) -> list[_TrackedWaypoint]:
    """Build the initial tracking list from PIHWaypoints (excludes start / approach_land)."""
    result = []
    for name in _WP_ORDER:
        xy_s = getattr(wpts, name)
        wx, wy, _ = xy_s
        if not math.isnan(wx):
            result.append(_TrackedWaypoint(
                name, wx, wy, _WP_THRESHOLDS[name],
                vy_min=_WP_VY_MIN.get(name, -float("inf")),
            ))
    return result


class PackageInHoleKTOController:
    """Tracks the PIH plan using the PD+feedforward controller from kto_lander.

    Phase 2 additions:
      - Waypoint index tracking (spatial proximity; advances when lander enters
        threshold distance of the current target waypoint).
      - Configurable replan triggers: "on_contact" and/or "periodic".
      - Oracle replanning via oracle_plan_pih_with_kto() + C2 weld.

    Parameters
    ----------
    plan, waypoints, cfg, params, gains:
        Same as before.
    replan_triggers:
        List of trigger names to enable: "on_contact", "periodic".
        Default [] → Phase 1 behaviour (no replan).
    replan_interval_s:
        Period for "periodic" trigger (seconds of simulation time).
    weld_strategy:
        Passed to weld_c2(); one of "point", "overlap_1", "overlap_2".
    """

    def __init__(
        self,
        plan: Plan,
        waypoints: PIHWaypoints,
        cfg: PIHConfig,
        params: LanderParams,
        gains: TrackerGains | None = None,
        *,
        replan_triggers: list[str] | None = None,
        replan_interval_s: float = 5.0,
        weld_strategy: str = "point",
    ):
        self.plan      = plan
        self.waypoints = waypoints
        self.cfg       = cfg
        self.params    = dataclasses.replace(params)
        self.gains     = gains if gains is not None else TrackerGains()
        self.t         = 0.0               # total elapsed simulation time
        self._saw_attachment = False
        self._contact_replan_pending = False  # True after attachment, until vy > 0
        self._pre_attach_params = dataclasses.replace(params)  # saved for oracle mass fix

        # Active plan tracking (switches on replan)
        self._active_plan   = plan
        self._active_plan_t = 0.0          # time within the current active plan
        landing_ref = plan(plan.T)
        self._hover_plan = _HoverPlan(float(landing_ref[0]), float(landing_ref[1]))

        # Waypoint index tracking
        self._tracked_waypoints: list[_TrackedWaypoint] = _waypoints_from_pih(waypoints)
        self._wp_idx: int = 0

        # Replan trigger config
        self.replan_triggers  = list(replan_triggers or [])
        self.replan_interval_s = replan_interval_s
        self.weld_strategy    = weld_strategy
        self._last_replan_t   = -float("inf")

        # Data collection: replan events + waypoint crossing states
        self.replan_events: list[dict] = []
        self.waypoint_states: dict[str, list[float]] = {}  # name → state when crossed

    # ── Public helpers ──────────────────────────────────────────────────

    @property
    def wp_idx(self) -> int:
        return self._wp_idx

    @property
    def current_waypoint_name(self) -> str | None:
        if self._wp_idx < len(self._tracked_waypoints):
            return self._tracked_waypoints[self._wp_idx].name
        return None

    def remaining_waypoint_names(self) -> list[str]:
        """Names of waypoints from current index onward."""
        return [w.name for w in self._tracked_waypoints[self._wp_idx:]]

    # ── Core step ──────────────────────────────────────────────────────

    def step(self, env) -> tuple[np.ndarray, dict]:
        """Compute one action from live Box2D state; advance internal clock by DT.

        Must be called exactly once per env.step() call.
        Returns (action[2], debug_dict).
        """
        uw = env.unwrapped

        # First call: inject rendering state
        if uw._kto_path_xy is None:
            T = self.plan.T
            uw._kto_path_xy = [
                (float(self.plan(s * T)[0]), float(self.plan(s * T)[1]))
                for s in np.linspace(0.0, 1.0, 80)
            ]
            uw._waypoints = self.waypoints
            uw._plan_ref  = self.plan

        uw._ctrl_t = self.t
        if uw._oracle_plan_ref is not None:
            uw._oracle_ctrl_t = self._active_plan_t

        lander = uw.lander
        pos    = lander.position
        vel    = lander.linearVelocity
        state  = np.array([
            float(pos.x), float(pos.y),
            float(vel.x), float(vel.y),
            float(lander.angle), float(lander.angularVelocity),
        ], dtype=np.float64)

        # ── Update mass on first attachment ────────────────────────────
        if uw._attached and not self._saw_attachment:
            self._saw_attachment = True
            d = PIH_LEG_OFFSET + self.cfg.package_height_assumed / 2.0
            self.params = dataclasses.replace(
                self.params,
                mass=self.params.mass + self.cfg.package_mass_assumed,
                inertia=self.params.inertia + self.cfg.package_mass_assumed * d * d,
            )
            if "on_contact" in self.replan_triggers:
                self._contact_replan_pending = True  # defer until lander reverses

        # Fire on_contact replan once lander is ascending after attachment bounce.
        # Replanning from vy < 0 (still descending at contact) causes the oracle
        # plan to start by going DOWN, which immediately diverges from the actual
        # lander trajectory after the contact bounce.
        if self._contact_replan_pending and state[3] > 0.0:
            self._contact_replan_pending = False
            self._fire_replan(state, trigger="on_contact", uw=uw)

        # ── Periodic replan ────────────────────────────────────────────
        if ("periodic" in self.replan_triggers
                and self.t - self._last_replan_t >= self.replan_interval_s):
            self._fire_replan(state, trigger="periodic", uw=uw)

        # ── Advance waypoint index by spatial proximity ────────────────
        self._advance_wp_idx(state)

        # ── Compute action from active plan ────────────────────────────
        if self._active_plan_t >= self._active_plan.T:
            action, debug = control(state, 0.0, self._hover_plan, self.params, self.gains)
        else:
            action, debug = control(state, self._active_plan_t, self._active_plan,
                                    self.params, self.gains)

        self.t              += DT
        self._active_plan_t += DT
        return action, debug

    # ── Internal helpers ────────────────────────────────────────────────

    def _advance_wp_idx(self, state: np.ndarray) -> None:
        """Advance _wp_idx when lander enters proximity of current target waypoint.

        Uses spatial proximity only (not temporal).  The vy_min guard on
        'extraction' prevents a false advance while the lander is descending
        through the extraction zone on its way down to contact.
        """
        while self._wp_idx < len(self._tracked_waypoints):
            wp = self._tracked_waypoints[self._wp_idx]
            dist = math.hypot(state[0] - wp.x, state[1] - wp.y)
            if dist < wp.threshold and state[3] >= wp.vy_min:
                self.waypoint_states[wp.name] = state.tolist()
                self._wp_idx += 1
            else:
                break

    def _fire_replan(self, state: np.ndarray, trigger: str, uw=None) -> None:
        """Run oracle KTO + C2 weld and switch to the welded plan."""
        self._last_replan_t = self.t

        # Remaining waypoints: skip "contact" if we just attached there
        remaining = self.remaining_waypoint_names()
        if trigger == "on_contact" and remaining and remaining[0] == "contact":
            remaining = remaining[1:]
        if not remaining:
            return  # nothing left to replan toward

        try:
            oracle_plan, oracle_T, oracle_wpts = oracle_plan_pih_with_kto(
                state, self.cfg, remaining, verbose=False,
            )
        except RuntimeError:
            return  # oracle solve failed; keep current plan

        # Oracle plan is already solved from actual state — switch directly.
        # Welding from old plan's state at weld_param causes position/velocity
        # mismatch (old plan descending; lander may already be ascending after
        # contact), which leads to x-deviation and extraction_collision.
        oracle_cps, oracle_knots = _plan_to_numpy(oracle_plan)

        # Fix tracker params to use oracle (true) mass/inertia.
        # Before attachment: params = lander-only.
        # After assumed-mass attachment: params = lander + assumed_package (may differ
        # from true). Oracle plan was solved with true mass, so tracking must use true.
        if self._saw_attachment:
            oracle_cfg = self.cfg.with_true_as_assumed()
            d = PIH_LEG_OFFSET + oracle_cfg.package_height_assumed / 2.0
            self.params = dataclasses.replace(
                self._pre_attach_params,
                mass=self._pre_attach_params.mass + oracle_cfg.package_mass_assumed,
                inertia=self._pre_attach_params.inertia + oracle_cfg.package_mass_assumed * d * d,
            )

        # Switch active plan
        self._active_plan   = oracle_plan
        self._active_plan_t = 0.0
        landing_ref = oracle_plan(oracle_plan.T)
        self._hover_plan = _HoverPlan(float(landing_ref[0]), float(landing_ref[1]))

        # Inject oracle arc + reference into renderer
        if uw is not None:
            uw._oracle_path_xy = [
                (float(oracle_plan(s * oracle_T)[0]), float(oracle_plan(s * oracle_T)[1]))
                for s in np.linspace(0.0, 1.0, 80)
            ]
            uw._oracle_plan_ref = oracle_plan
            uw._oracle_ctrl_t   = 0.0

        # Update tracked waypoints to oracle's waypoints
        new_tracked = _waypoints_from_pih(oracle_wpts)
        if new_tracked:
            self._tracked_waypoints = new_tracked
            self._wp_idx = 0

        # Record the event (include oracle CPs for data analysis)
        self.replan_events.append({
            "trigger":          trigger,
            "sim_time":         self.t,
            "state_at_trigger": state.tolist(),
            "remaining_wpts":   remaining,
            "oracle_T":         oracle_T,
            "oracle_wpts":      oracle_wpts.to_dict(),
            "oracle_cps":       oracle_cps.tolist(),
            "oracle_knots":     oracle_knots.tolist(),
        })
