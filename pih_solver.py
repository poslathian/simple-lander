"""PIH solver: KTO trajectory planner and tracking controller for PackageInHoleEnv."""

from __future__ import annotations

import dataclasses

import numpy as np
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import LinearEqualityConstraint, Solve

from kto_lander import Plan, LanderParams, TrackerGains, control
from pih_env import (
    PIHConfig,
    PIH_START_X, PIH_PICKUP_X, PIH_PAD_Y, PIH_LEG_OFFSET,
    PIH_MOUNTAIN_X, PIH_MOUNTAIN_H,
    DT,
)

# Clearance above the mountain peak (m).  Must exceed body height plus PD
# tracking error headroom during the crossing.
_MOUNTAIN_CLEARANCE_M = 1.5

# Default normalised times for the 7 PIH waypoints (approach_land is a read-back annotation).
# These are the primary tuning knob as the task grows more complex.
_DEFAULT_S: tuple[float, ...] = (0.00, 0.20, 0.40, 0.52, 0.62, 0.77, 0.88, 1.00)


@dataclasses.dataclass(frozen=True)
class PIHWaypoints:
    """Named milestones along the PIH plan, each as (x_world, y_world, s_normalised).

    s ∈ [0, 1] is the normalised KTO time; multiply by plan.T to get wall-clock
    seconds.  All coordinates are absolute world metres (not pad-centred).

    The normalised times are the primary tuning knob as the task grows more complex —
    pass an explicit waypoints_s to plan_pih_with_kto() to override them.
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


def plan_pih_with_kto(
    x0: float,
    y0: float,
    vx0: float,
    vy0: float,
    cfg: PIHConfig,
    params: LanderParams,
    *,
    waypoints_s: tuple[float, ...] | None = None,
    num_control_points: int = 20,
    spline_order: int = 4,
    T_min: float = 10.0,
    T_max: float = 30.0,
    vy_contact: float = -0.1,
    vy_extraction: float = 0.5,
    vy_landing: float = -0.3,
    verbose: bool = True,
) -> tuple[Plan, float, PIHWaypoints]:
    """Solve a PIH trajectory via pydrake KTO.

    Works in absolute world coordinates because the task has two pads.
    control() from kto_lander is coordinate-agnostic and accepts an
    absolute-coordinate plan paired with absolute state directly.

    Returns (Plan, duration_T_seconds, PIHWaypoints).
    Raises RuntimeError on solver failure.
    """
    if waypoints_s is None:
        waypoints_s = _DEFAULT_S
    if len(waypoints_s) != 8:
        raise ValueError(f"waypoints_s must have 8 entries, got {len(waypoints_s)}")
    s0, s_mo, s_ap, s_ct, s_ex, s_mr, s_al, s1 = waypoints_s

    # Waypoint positions — planner only sees assumed geometry
    mountain_y_out  = PIH_PAD_Y + PIH_MOUNTAIN_H + _MOUNTAIN_CLEARANCE_M + 0.5  # extra 0.5 m headroom
    # On return the package hangs below the lander by ~LEG_OFFSET + package height,
    # so raise mountain_ret by package_height_assumed to keep it clear of the peak.
    mountain_y_ret  = mountain_y_out + cfg.package_height_assumed
    approach_y      = cfg.assumed_contact_lander_y + 2.5               # ~7.6 m (pickup)
    landing_y = PIH_PAD_Y + PIH_LEG_OFFSET - 0.15                       # ~4.45 m

    # approach_land has no prescribed position — its actual position is read back
    # from the spline after solving and stored in the waypoints for rendering.
    wpts_fixed = dict(
        start         = (x0,             y0,                            s0),
        mountain_out  = (PIH_MOUNTAIN_X, mountain_y_out,                s_mo),
        approach      = (PIH_PICKUP_X,   approach_y,                    s_ap),
        contact       = (PIH_PICKUP_X,   cfg.assumed_contact_lander_y,  s_ct),
        extraction    = (PIH_PICKUP_X,   cfg.extraction_lander_y,       s_ex),
        mountain_ret  = (PIH_MOUNTAIN_X, mountain_y_ret,                s_mr),
        landing       = (PIH_START_X,    landing_y,                     s1),
    )

    # Conservative acceleration / velocity box — same derivation as kto_lander.py.
    # ax_max=1.5 ⟹ worst-case flatness atan2(1.5,9)=0.165 rad ≈ 9.5° — tighter than
    # the old 3.0 (18°) to prevent angular-rate spikes in the descent arc.
    ax_max, ay_min, ay_max = 1.5, -1.0, 5.0
    vx_max, vy_min, vy_max = 7.0, -7.0, 5.0

    kto = KinematicTrajectoryOptimization(
        num_positions=2,
        num_control_points=num_control_points,
        spline_order=spline_order,
    )
    kto.AddDurationConstraint(T_min, T_max)
    kto.AddDurationCost(1.0)
    kto.AddPathLengthCost(0.02)
    kto.AddVelocityBounds(np.array([-vx_max, vy_min]), np.array([vx_max, vy_max]))
    kto.AddAccelerationBounds(np.array([-ax_max, ay_min]), np.array([ax_max, ay_max]))

    # Position constraints at the 7 prescribed waypoints
    for wx, wy, ws in wpts_fixed.values():
        kto.AddPathPositionConstraint(np.array([wx, wy]), np.array([wx, wy]), ws)

    # Zero acceleration at boundaries → flatness gives θ_ref ≈ 0 at start and landing
    zero2 = np.zeros(2)
    kto.AddPathAccelerationConstraint(zero2, zero2, s0)
    kto.AddPathAccelerationConstraint(zero2, zero2, s1)

    def _vel_eq(vx_req: float, vy_req: float) -> LinearEqualityConstraint:
        A = np.zeros((2, 4))
        A[0, 2] = 1.0  # qdot_x
        A[1, 3] = 1.0  # qdot_y
        return LinearEqualityConstraint(A, np.array([vx_req, vy_req]))

    kto.AddVelocityConstraintAtNormalizedTime(_vel_eq(vx0, vy0),           s0)    # initial
    kto.AddVelocityConstraintAtNormalizedTime(_vel_eq(0.0, vy_contact),    s_ct)  # hover at contact
    kto.AddVelocityConstraintAtNormalizedTime(_vel_eq(0.0, vy_extraction), s_ex)  # ascend at extraction
    kto.AddVelocityConstraintAtNormalizedTime(_vel_eq(0.0, vy_landing),    s1)    # settle at landing

    result = Solve(kto.prog())
    if not result.is_success():
        if verbose:
            print(f"[pih_kto] FAILED: {result.get_solver_id().name()} "
                  f"{result.get_solution_result()}")
        raise RuntimeError(f"PIH KTO solve failed: {result.get_solution_result()}")

    traj = kto.ReconstructTrajectory(result)
    T = float(traj.end_time() - traj.start_time())

    # Read the actual approach_land position from the solved trajectory
    al_q = traj.value(s_al * T).flatten()
    wpts = PIHWaypoints(
        approach_land=(float(al_q[0]), float(al_q[1]), s_al),
        **wpts_fixed,
    )

    if verbose:
        print(f"[pih_kto] solved: T={T:.2f}s  #cp={num_control_points}  "
              f"solver={result.get_solver_id().name()}")
        for f in dataclasses.fields(wpts):
            wx, wy, ws = getattr(wpts, f.name)
            print(f"  {f.name:<14s} ({wx:6.2f}, {wy:5.2f})  s={ws:.2f}  t={ws * T:5.1f}s")

    return Plan(traj, T), T, wpts


class _HoverPlan:
    """Fake plan that holds the terminal landing position with zero velocity/accel.

    Used after the KTO plan ends so the PD controller targets pure hover rather
    than the vy=-0.3 terminal velocity, eliminating the downward drift bias.
    """

    def __init__(self, landing_x: float, landing_y: float) -> None:
        self.T = 0.0
        self._ref = (landing_x, landing_y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __call__(self, t: float):  # noqa: ANN201
        return self._ref


class PackageInHoleKTOController:
    """Tracks the PIH plan using the PD+feedforward controller from kto_lander.

    State is read directly from Box2D in absolute world coordinates, matching
    the absolute-coordinate plan produced by plan_pih_with_kto().

    On the first step after env._attached becomes True, params.mass is increased
    by cfg.package_mass_assumed so the controller accounts for the payload.
    """

    def __init__(
        self,
        plan: Plan,
        waypoints: PIHWaypoints,
        cfg: PIHConfig,
        params: LanderParams,
        gains: TrackerGains | None = None,
    ):
        self.plan      = plan
        self.waypoints = waypoints           # stored for diagnostics / tuning interrogation
        self.cfg       = cfg
        self.params    = dataclasses.replace(params)  # local mutable copy
        self.gains     = gains if gains is not None else TrackerGains()
        self.t         = 0.0
        self._saw_attachment = False
        # Pure-hover plan used after trajectory ends, eliminating the vy=-0.3 terminal
        # bias that would keep the controller pushing downward after the KTO plan ends.
        landing_ref = plan(plan.T)
        self._hover_plan = _HoverPlan(float(landing_ref[0]), float(landing_ref[1]))

    def step(self, env) -> tuple[np.ndarray, dict]:
        """Compute one action from live Box2D state; advance internal clock by DT.

        Must be called exactly once per env.step() call.
        Returns (action[2], debug_dict).

        Also auto-injects rendering state into env.unwrapped on the first call
        (_kto_path_xy, _waypoints, _plan_ref) and updates _ctrl_t every call.
        """
        uw = env.unwrapped

        # First call: sample the trajectory and store rendering references
        if uw._kto_path_xy is None:
            T = self.plan.T
            uw._kto_path_xy = [
                (float(self.plan(s * T)[0]), float(self.plan(s * T)[1]))
                for s in np.linspace(0.0, 1.0, 80)
            ]
            uw._waypoints = self.waypoints
            uw._plan_ref  = self.plan

        uw._ctrl_t = self.t

        lander = uw.lander
        pos    = lander.position
        vel    = lander.linearVelocity
        state  = np.array([
            float(pos.x),        float(pos.y),
            float(vel.x),        float(vel.y),
            float(lander.angle), float(lander.angularVelocity),
        ], dtype=np.float64)

        # Update mass + inertia the first time attachment is detected.
        # Inertia uses parallel-axis theorem: I += m_pkg * d^2 where d is the
        # distance from lander COM to package COM (leg offset + half package height).
        if uw._attached and not self._saw_attachment:
            self._saw_attachment = True
            d = PIH_LEG_OFFSET + self.cfg.package_height_assumed / 2.0
            self.params = dataclasses.replace(
                self.params,
                mass=self.params.mass + self.cfg.package_mass_assumed,
                inertia=self.params.inertia + self.cfg.package_mass_assumed * d * d,
            )

        if self.t >= self.plan.T:
            action, debug = control(state, 0.0, self._hover_plan, self.params, self.gains)
        else:
            action, debug = control(state, self.t, self.plan, self.params, self.gains)
        self.t += DT
        return action, debug
