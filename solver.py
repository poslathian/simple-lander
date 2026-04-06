"""Open-loop lunar lander trajectory optimization using Drake's KTO.

Uses KinematicTrajectoryOptimization to find a smooth [x, y, theta] B-spline
trajectory from a start position down to the landing pad.  Nonlinear dynamics
constraints at sampled times ensure every acceleration along the path is
achievable within the lander's thrust limits.  Circular obstacles are avoided
via sampled distance constraints.

Physics model (2D rigid-body rocket, constants from lunar_lander.py):

    Main engine force direction (body up):  (-sin θ,  cos θ)
    Side engine force direction (body right): ( cos θ, -sin θ)

    x''  = (-Fm sin θ  +  Fs cos θ) / m
    y''  = ( Fm cos θ  -  Fs sin θ) / m  -  g

Given (x'', y'', θ) we can invert for the required thrusts:

    Fm =  m * ( -x'' sin θ  + (y'' + g) cos θ )
    Fs =  m * (  x'' cos θ  - (y'' + g) sin θ )

Torque from the side engine applied at the Box2D impulse point:

    τ  = Fs · (2·A_AWAY·sinθ·cosθ + ARM_A·sin²θ - ARM_B·cos²θ) / I

where A_AWAY = SIDE_ENGINE_AWAY / SCALE (lateral offset of application point).
"""

from dataclasses import dataclass
import time
import numpy as np
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import Solve, SnoptSolver, SolverOptions
from pydrake.trajectories import BsplineTrajectory
from scipy.interpolate import BSpline

import lunar_lander as ll

# ── World geometry (matches Gymnasium LunarLander viewport) ──────────────
SCALE = ll.SCALE
VIEWPORT_W, VIEWPORT_H = ll.VIEWPORT_W, ll.VIEWPORT_H
W = VIEWPORT_W / SCALE  # 30.0  world units
H = VIEWPORT_H / SCALE  # 20.0  world units
PAD_X = W / 2            # 15.0
PAD_Y = H / 4            # 5.0

# ── 2D rocket physics (imported from lunar_lander surrogate) ───────────
GRAVITY = abs(ll.GRAVITY)                    # 10.0
MASS = ll.LANDER_BODY_MASS                   # ~4.817 (lander body only; legs massless in flight)
INERTIA = ll.LANDER_INERTIA_CM              # ~0.833

# Max continuous force = peak impulse / dt
THRUST_MAX = (ll.MAIN_ENGINE_POWER
              * ll.MAIN_ENGINE_Y_LOCATION / (ll.SCALE * ll.DT))
SIDE_MAX   = (ll.SIDE_ENGINE_POWER
              * ll.SIDE_ENGINE_AWAY / (ll.SCALE * ll.DT))

# Re-export torque arm constants from the canonical source in lunar_lander
SIDE_ARM_A = ll.SIDE_ARM_A
SIDE_ARM_B = ll.SIDE_ARM_B
SIDE_AWAY  = ll.SIDE_AWAY_SCALED

# ── Default boundary conditions ──────────────────────────────────────────
START = np.array([PAD_X + 4.0, H - 0.5, 0.0])
GOAL = np.array([PAD_X, PAD_Y, 0.0])

SPLINE_ORDER = 4  # cubic B-spline


# ─────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────

def _n_unique_knots(n_cp, order=SPLINE_ORDER):
    """Number of unique knot values for a clamped uniform B-spline."""
    return n_cp - order + 2


@dataclass
class Strategy:
    """Encapsulates all solver formulation decisions.

    Warm-start phase always uses a point goal constraint and zero end
    velocity.  The obstacle phase can relax the goal to a region and
    add a quadratic cost to shape the gradient toward the goal pose.

    Attributes
    ----------
    name : str
        Human-readable label.
    warmstart_frac : float
        Fraction of the total iteration budget allocated to warm-start.
        Remainder goes to the obstacle phase.
    goal_region : bool
        If True, the obstacle phase relaxes the goal to a box constraint
        (pad width ±2, theta ±0.4) instead of an exact pose.
    goal_cost_weight : float
        Weight on quadratic cost pulling the endpoint toward the goal
        pose.  Only meaningful when goal_region is True.
    energy_cost : float
        Weight on AddPathEnergyCost.  Smooths the trajectory but can
        fight obstacle avoidance.  0 disables.
    duration_cost : float
        Weight on AddDurationCost (minimize time).
    num_control_points : int
        B-spline control points.
    constraint_scale : float
        Dynamics/obstacle constraint density relative to unique knots.
        1.0 = one sample per unique knot.  2.0 = sample at and between
        every knot (2× density).
    """
    name: str = "default"
    warmstart_frac: float = 0.0     # 0 = uncapped warm-start
    goal_region: bool = False
    goal_cost_weight: float = 0.0
    energy_cost: float = 1.0
    duration_cost: float = 1.0
    num_control_points: int = 10
    constraint_scale: float = 5.0

    @property
    def num_dynamics_samples(self):
        n_knots = _n_unique_knots(self.num_control_points)
        return max(3, int(n_knots * self.constraint_scale))


STRATEGIES = {
    "default": Strategy(
        name="default",
        warmstart_frac=0.0,
        goal_region=False,
        goal_cost_weight=0.0,
        energy_cost=1.0,
        duration_cost=5.0,
        num_control_points=15,
    ),
    "region_goal": Strategy(
        name="region_goal",
        warmstart_frac=0.4,
        goal_region=True,
        goal_cost_weight=50.0,
        energy_cost=0.0,
        duration_cost=1.0,
    ),
}


# ─────────────────────────────────────────────────────────────────────────
# B-spline evaluation
# ─────────────────────────────────────────────────────────────────────────

def _basis_weights(basis, s, deriv=0):
    """Evaluate each B-spline basis function (or its k-th derivative) at s.

    Returns a length-N weight vector w such that  r^(k)(s) = control_points @ w.
    """
    knots = np.array(basis.knots())
    degree = basis.order() - 1
    n = basis.num_basis_functions()
    w = np.zeros(n)
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1.0
        b = BSpline(knots, c, degree)
        if deriv:
            b = b.derivative(deriv)
        w[i] = float(b(s))
    return w


def _sample_xy(traj, n=300):
    """Sample [x,y] positions along a trajectory."""
    times = np.linspace(traj.start_time(), traj.end_time(), n)
    xy = np.empty((n, 2))
    for i, t in enumerate(times):
        xy[i] = traj.value(t).flatten()[:2]
    return xy


# ─────────────────────────────────────────────────────────────────────────
# Trajectory optimization
# ─────────────────────────────────────────────────────────────────────────

def solve(start=None, goal=None, obstacles=(),
          terrain=None,
          on_progress=None, max_iters=None,
          time_budget=0.25, warmstart_budget=0.10,
          strategy=None):
    """Plan a minimum-time landing trajectory.

    Parameters
    ----------
    start, goal : array-like (3,), optional
        [x, y, theta].  Defaults to module-level START / GOAL.
    obstacles : sequence of (cx, cy, radius) tuples
        Circular no-fly zones the trajectory must avoid.
    terrain : (txs, tys) tuple or None
        Terrain x/y arrays for ground-avoidance constraints.
        If None, only the global y-lower-bound is enforced.
    on_progress : callable(phase_str, frac) or None
        Called at milestones so callers can update a UI.
    max_iters : int or None
        If set, use fixed iteration budget (legacy).  Otherwise use
        time_budget / warmstart_budget wall-clock limits.
    time_budget : float
        Total wall-clock seconds for both solver phases (default 1.0).
    warmstart_budget : float
        Max wall-clock seconds for warm-start phase (default 0.33).
        Unused warm-start time is donated to the obstacle phase.
    strategy : Strategy or str or None
        Solver formulation strategy.  Pass a name from STRATEGIES,
        a Strategy instance, or None for the default.

    Returns
    -------
    times : ndarray
    plan : dict of ndarrays
    constraint_xy : (n_dyn, 2) ndarray
    warm_xy : (300, 2) ndarray — the warm-start (obstacle-free) path
    knot_xy : (n_knots, 2) ndarray — positions at B-spline knots
    control_xy : (n_cp, 2) ndarray — B-spline control point positions
    """
    start = np.asarray(start if start is not None else START, dtype=float)
    goal = np.asarray(goal if goal is not None else GOAL, dtype=float)

    if strategy is None:
        strat = STRATEGIES["default"]
    elif isinstance(strategy, str):
        strat = STRATEGIES[strategy]
    else:
        strat = strategy

    n_cp = strat.num_control_points
    n_dyn = strat.num_dynamics_samples
    has_hard_constraints = bool(obstacles) or (terrain is not None)

    # Iteration chunk size for time-budgeted solving
    CHUNK = 10

    def _report(phase, frac):
        if on_progress:
            on_progress(phase, frac)

    def _build(initial_guess_traj=None, phase="warmstart"):
        """Build a fresh KTO + MathematicalProgram."""
        kto = KinematicTrajectoryOptimization(
            num_positions=3,
            num_control_points=n_cp,
            spline_order=SPLINE_ORDER,
            duration=3.0,
        )
        prog = kto.get_mutable_prog()

        # Start: always exact
        kto.AddPathPositionConstraint(start, start, 0.0)

        # Goal: exact for warm-start, optionally relaxed for obstacle phase
        if phase == "obstacle" and strat.goal_region:
            goal_lb = np.array([PAD_X - 2.0, goal[1], -0.4])
            goal_ub = np.array([PAD_X + 2.0, goal[1],  0.4])
            kto.AddPathPositionConstraint(goal_lb, goal_ub, 1.0)
        else:
            kto.AddPathPositionConstraint(goal, goal, 1.0)

        # Velocity: always zero at endpoints
        z = np.zeros((3, 1))
        kto.AddPathVelocityConstraint(z, z, 0.0)
        kto.AddPathVelocityConstraint(z, z, 1.0)

        kto.AddPositionBounds(
            np.array([0.0, PAD_Y - 1.0, -np.pi / 3]),
            np.array([W,   H + 1.0,      np.pi / 3]),
        )
        kto.AddVelocityBounds(
            np.array([-5.0, -10.0, -3.0]),
            np.array([ 5.0,   1.0,  3.0]),
        )
        a_max = THRUST_MAX / MASS + GRAVITY
        alpha_max = SIDE_MAX * max(SIDE_ARM_A, SIDE_ARM_B) / INERTIA
        kto.AddAccelerationBounds(
            np.array([-a_max, -a_max, -alpha_max]),
            np.array([ a_max,  a_max,  alpha_max]),
        )
        kto.AddDurationConstraint(1.5, 5.0)

        # Costs
        if strat.duration_cost > 0:
            kto.AddDurationCost(strat.duration_cost)
        if strat.energy_cost > 0:
            kto.AddPathEnergyCost(strat.energy_cost)

        # Quadratic goal cost (pulls endpoint toward exact goal pose)
        if phase == "obstacle" and strat.goal_region and strat.goal_cost_weight > 0:
            _add_goal_cost(kto, prog, goal, strat.goal_cost_weight)

        if initial_guess_traj is not None:
            kto.SetInitialGuess(initial_guess_traj)
        else:
            nc = kto.num_control_points()
            cp0 = np.column_stack(
                [start + (goal - start) * t for t in np.linspace(0, 1, nc)]
            )
            kto.SetInitialGuess(BsplineTrajectory(kto.basis(), cp0))

        _add_dynamics_constraints(kto, prog, n_dyn)
        if phase != "warmstart":
            _add_terrain_constraints(kto, prog, terrain, n_dyn)
        return kto, prog

    # ── Phase 1: warm-start (dynamics only, no obstacles/terrain) ────────
    _report("building", 0.0)
    kto, prog = _build(phase="warmstart")
    _report("warm-start", 0.2)

    t_start = time.monotonic()
    if max_iters is not None:
        # Legacy fixed-iteration mode
        ws_limit = max(2, int(max_iters * 0.3)) if has_hard_constraints else max_iters
        opts1 = SolverOptions()
        opts1.SetOption(SnoptSolver.id(), "Major iterations limit", ws_limit)
        result = Solve(prog, solver_options=opts1)
    else:
        # Time-budgeted: run in chunks until converged or budget exhausted
        ws_deadline = t_start + warmstart_budget
        result = None
        total_ws_iters = 0
        while time.monotonic() < ws_deadline:
            opts1 = SolverOptions()
            opts1.SetOption(SnoptSolver.id(), "Major iterations limit", CHUNK)
            result = Solve(prog, solver_options=opts1)
            total_ws_iters += CHUNK
            if result.is_success():
                break

    warm_traj = kto.ReconstructTrajectory(result)
    warm_xy = _sample_xy(warm_traj)
    t_after_ws = time.monotonic()

    if not has_hard_constraints:
        _report("sampling", 0.9)
        times, plan, cpts, knot_xy, control_xy, cp3d, dur = _sample(
            warm_traj, n_constraint_pts=n_dyn)
        _report("done", 1.0)
        return times, plan, cpts, warm_xy, knot_xy, control_xy, cp3d, dur

    # ── Phase 2: re-solve with obstacles and terrain ───────────────────
    _report("obstacles", 0.4)
    kto2, prog2 = _build(initial_guess_traj=warm_traj, phase="obstacle")
    _add_obstacle_constraints(kto2, prog2, obstacles, n_dyn)

    _report("solving", 0.6)
    if max_iters is not None:
        # Legacy fixed-iteration mode
        obs_limit = max(2, max_iters - int(max_iters * 0.3))
        opts2 = SolverOptions()
        opts2.SetOption(SnoptSolver.id(), "Major iterations limit", obs_limit)
        result2 = Solve(prog2, solver_options=opts2)
    else:
        # Time-budgeted: use all remaining time from the total budget
        # (includes any time saved by early warm-start convergence)
        obs_deadline = t_start + time_budget
        result2 = None
        total_obs_iters = 0
        while time.monotonic() < obs_deadline:
            opts2 = SolverOptions()
            opts2.SetOption(SnoptSolver.id(), "Major iterations limit", CHUNK)
            result2 = Solve(prog2, solver_options=opts2)
            total_obs_iters += CHUNK
            if result2.is_success():
                break

    best_traj = kto2.ReconstructTrajectory(result2)

    if not result2.is_success():
        p2_end = best_traj.value(best_traj.end_time()).flatten()
        goal_err = np.linalg.norm(p2_end - goal)
        if goal_err > 1.0:
            print("WARNING: obstacle solve diverged, using warm-start plan")
            best_traj = warm_traj
        else:
            iters_str = total_obs_iters if max_iters is None else obs_limit
            print(f"NOTE: obstacle solve used {iters_str} iters "
                  f"(best-effort, goal err={goal_err:.3f})")

    _report("sampling", 0.9)
    times, plan, cpts, knot_xy, control_xy, cp3d, dur = _sample(
        best_traj, n_constraint_pts=n_dyn)
    _report("done", 1.0)
    return times, plan, cpts, warm_xy, knot_xy, control_xy, cp3d, dur


# ─────────────────────────────────────────────────────────────────────────
# Constraint & cost helpers
# ─────────────────────────────────────────────────────────────────────────

def _add_goal_cost(kto, prog, goal, weight):
    """Add quadratic cost on endpoint deviation from goal pose.

    cost = weight * || pos(s=1) - goal ||^2

    Built as a proper quadratic cost (Q matrix + b vector) so Drake
    can compute exact gradients without AutoDiff callbacks.
    """
    basis = kto.basis()
    n_cp = kto.num_control_points()
    cp = kto.control_points()  # (3, n_cp) decision variable matrix
    w = _basis_weights(basis, 1.0, deriv=0)

    # pos(1) = cp @ w, so err_i = sum_j cp[i,j]*w[j] - goal[i]
    # ||err||^2 = sum_i (sum_j cp[i,j]*w[j] - goal[i])^2
    # This is quadratic in the cp variables.
    for i in range(3):
        # Variables for row i of the control point matrix
        row_vars = cp[i, :]
        # Q = weight * w @ w^T,  b = -2 * weight * goal[i] * w
        Q = 2.0 * weight * np.outer(w, w)
        b = -2.0 * weight * goal[i] * w
        c = weight * goal[i] ** 2
        prog.AddQuadraticCost(Q, b, c, row_vars)


def _validate_torque_model():
    """Verify the analytical torque formula matches lander_acceleration().

    Called at import time.  If the impulse geometry in lunar_lander.py changes,
    this will raise AssertionError, forcing the solver's dynamics model to be
    updated to match.
    """
    test_thetas = [0.0, 0.3, -0.5, np.pi / 4, -np.pi / 6]
    test_sides = [-0.8, -0.3, 0.3, 0.7]

    for theta in test_thetas:
        state = [10.0, 10.0, theta, 0.0, 0.0, 0.0]
        # Main engine should produce zero angular acceleration
        ax_m, ay_m, alpha_m = ll.lander_acceleration(state, THRUST_MAX * 0.9, 0.0)
        assert abs(alpha_m) < 1e-6, (
            f"Main engine torque non-zero at θ={theta}: {alpha_m}")

        ct, st = np.cos(theta), np.sin(theta)
        torque_arm = (2 * SIDE_AWAY * st * ct
                      + SIDE_ARM_A * st * st
                      - SIDE_ARM_B * ct * ct)
        for s_frac in test_sides:
            Fs = SIDE_MAX * s_frac
            _, _, alpha_phys = ll.lander_acceleration(state, 0.0, Fs)
            alpha_formula = Fs * torque_arm / INERTIA
            assert abs(alpha_phys - alpha_formula) < 1e-6, (
                f"Torque mismatch at θ={theta}, Fs={Fs:.2f}: "
                f"physics={alpha_phys}, formula={alpha_formula}")

_validate_torque_model()


def _add_dynamics_constraints(kto, prog, n_samples):
    """Constrain implied thrusts to be within physical limits at each sample.

    Uses the lunar_lander.py torque model, validated at import time by
    _validate_torque_model().

    Force directions (world frame):
      Main: Fm · (-sinθ,  cosθ)   — body up
      Side: Fs · ( cosθ, -sinθ)   — body right

    Torque from side engine at Box2D application point:
      τ = Fs · (2·SIDE_AWAY·sinθ·cosθ + SIDE_ARM_A·sin²θ - SIDE_ARM_B·cos²θ)

    Outputs per sample:  [Fm, Fs, torque_error]
    Bounds:              [0, THRUST_MAX] x [-SIDE_MAX, SIDE_MAX] x {0}
    """
    cp = kto.control_points()
    T = kto.duration()
    basis = kto.basis()
    n_cp = kto.num_control_points()
    all_vars = np.concatenate([cp.flatten(), [T]])

    TORQUE_TOL = 0.001  # tight tolerance for accurate thrust replay
    lb = np.array([0.0,       -SIDE_MAX, -TORQUE_TOL])
    ub = np.array([THRUST_MAX, SIDE_MAX,  TORQUE_TOL])

    for s in np.linspace(0, 1, n_samples):
        w_pos = _basis_weights(basis, s, deriv=0)
        w_acc = _basis_weights(basis, s, deriv=2)

        def _make(w_pos, w_acc):
            def constraint(v):
                P = v[:-1].reshape(3, n_cp)
                dur = v[-1]
                pos = P @ w_pos
                acc = P @ w_acc / dur ** 2

                ct, st = np.cos(pos[2]), np.sin(pos[2])
                Fm = MASS * (-acc[0] * st + (acc[1] + GRAVITY) * ct)
                Fs = MASS * ( acc[0] * ct - (acc[1] + GRAVITY) * st)

                # Torque from side impulse at Box2D application point
                torque_arm = (2 * SIDE_AWAY * st * ct
                              + SIDE_ARM_A * st * st
                              - SIDE_ARM_B * ct * ct)
                torque_err = acc[2] - Fs * torque_arm / INERTIA
                return np.array([Fm, Fs, torque_err])
            return constraint

        prog.AddConstraint(_make(w_pos, w_acc), lb, ub, all_vars)


def _smooth_terrain_height(x_val, txs, tys):
    """AutoDiff-compatible piecewise-linear terrain interpolation.

    Uses soft hat functions (softplus approximation of ReLU) to blend
    terrain vertex values smoothly while staying close to np.interp.
    """
    k = 50.0  # softplus sharpness

    def _softplus(z):
        # log(1 + exp(k*z)) / k  ≈  max(0, z)  for large k
        return np.log(1.0 + np.exp(np.clip(k * z, -30.0, 30.0))) / k

    ground = 0.0
    w_total = 1e-12
    for i in range(len(txs)):
        # Hat function: phi_i(x) = max(0, 1 - |x - txs[i]| / dx)
        # Use softplus as smooth max(0, ...)
        if i > 0:
            dx_left = txs[i] - txs[i - 1]
        else:
            dx_left = txs[1] - txs[0]
        if i < len(txs) - 1:
            dx_right = txs[i + 1] - txs[i]
        else:
            dx_right = txs[-1] - txs[-2]

        # Two-sided hat: rises linearly from txs[i-1] to txs[i], falls to txs[i+1]
        left_val = 1.0 - (txs[i] - x_val) / dx_left
        right_val = 1.0 - (x_val - txs[i]) / dx_right
        # phi = min(left_val, right_val), clamped to [0, 1]
        # Use soft approximation: phi ≈ softplus(left) * softplus(right) / softplus(1)^2
        phi = _softplus(left_val) * _softplus(right_val)
        ground += phi * tys[i]
        w_total += phi
    return ground / w_total


def _add_terrain_constraints(kto, prog, terrain, n_samples):
    """Constrain y(s) >= terrain_height(x(s)) + margin at sample points."""
    if terrain is None:
        return
    txs, tys = np.asarray(terrain[0], dtype=float), np.asarray(terrain[1], dtype=float)
    basis = kto.basis()
    n_cp = kto.num_control_points()
    cp_flat = kto.control_points().flatten()
    TERRAIN_MARGIN = 0.5

    # Terrain is smooth — use fewer samples than dynamics constraints
    n_terrain = min(n_samples, 10)
    for s in np.linspace(0, 1, n_terrain):
        w = _basis_weights(basis, s, deriv=0)

        def _make(w, txs, tys):
            def constraint(v):
                P = v.reshape(3, n_cp)
                pos = P @ w
                ground = _smooth_terrain_height(pos[0], txs, tys)
                return np.array([pos[1] - ground - TERRAIN_MARGIN])
            return constraint

        prog.AddConstraint(
            _make(w, txs, tys),
            np.array([0.0]), np.array([np.inf]),
            cp_flat,
        )


def _add_obstacle_constraints(kto, prog, obstacles, n_samples):
    """For each obstacle, constrain  (x-cx)^2 + (y-cy)^2 >= (r+margin)^2
    at every sample point along the path.

    Uses squared distance (no sqrt) so the constraint stays smooth everywhere.
    Only depends on control-point variables (not duration).
    """
    if not obstacles:
        return
    basis = kto.basis()
    n_cp = kto.num_control_points()
    cp_flat = kto.control_points().flatten()
    # Margin = lander bounding circle so we avoid with the full rect, not just center
    LANDER_RADIUS = np.hypot(17.0 / SCALE, 22.0 / SCALE)  # ≈ 0.93
    MARGIN = LANDER_RADIUS

    # Cap obstacle samples — too many makes the problem intractable
    n_obs = min(n_samples, 20)
    for cx, cy, r in obstacles:
        r_eff = r + MARGIN
        for s in np.linspace(0, 1, n_obs):
            w = _basis_weights(basis, s, deriv=0)

            def _make(w, cx, cy, r_eff):
                def constraint(v):
                    P = v.reshape(3, n_cp)
                    pos = P @ w
                    dx, dy = pos[0] - cx, pos[1] - cy
                    return np.array([dx * dx + dy * dy - r_eff * r_eff])
                return constraint

            prog.AddConstraint(
                _make(w, cx, cy, r_eff),
                np.array([0.0]), np.array([np.inf]),
                cp_flat,
            )


# ─────────────────────────────────────────────────────────────────────────
# Trajectory sampling
# ─────────────────────────────────────────────────────────────────────────

def _sample(traj, n=300, n_constraint_pts=8):
    """Sample the solved trajectory, computing thrusts via inverse dynamics.

    Returns (times, plan_dict, constraint_xy, knot_xy, control_xy) where:
    - constraint_xy: (n_constraint_pts, 2) positions at dynamics sample points
    - knot_xy: (n_knots, 2) positions at unique B-spline knot times
    - control_xy: (n_cp, 2) B-spline control point positions
    """
    times = np.linspace(traj.start_time(), traj.end_time(), n)
    S = {k: np.empty(n) for k in
         ("x", "y", "theta", "vx", "vy", "omega", "Fm", "Fs",
          "ax", "ay", "alpha")}

    for i, t in enumerate(times):
        q   = traj.value(t).flatten()
        qd  = traj.EvalDerivative(t, 1).flatten()
        qdd = traj.EvalDerivative(t, 2).flatten()

        S["x"][i], S["y"][i], S["theta"][i] = q
        S["vx"][i], S["vy"][i], S["omega"][i] = qd
        S["ax"][i], S["ay"][i], S["alpha"][i] = qdd

        ct, st = np.cos(q[2]), np.sin(q[2])
        S["Fm"][i] = MASS * (-qdd[0] * st + (qdd[1] + GRAVITY) * ct)
        S["Fs"][i] = MASS * ( qdd[0] * ct - (qdd[1] + GRAVITY) * st)

    t0, t1 = traj.start_time(), traj.end_time()

    # Constraint enforcement points
    cpts = np.empty((n_constraint_pts, 2))
    for i, s in enumerate(np.linspace(0, 1, n_constraint_pts)):
        t = t0 + s * (t1 - t0)
        q = traj.value(t).flatten()
        cpts[i] = q[:2]

    # Knot positions (unique internal + boundary knots)
    basis = traj.basis()
    knots = np.array(basis.knots())
    unique_knots = np.unique(knots)
    knot_xy = np.empty((len(unique_knots), 2))
    for i, kt in enumerate(unique_knots):
        knot_xy[i] = traj.value(kt).flatten()[:2]

    # Control point positions (B-spline control points in x,y)
    cp_list = traj.control_points()  # list of (3,1) arrays
    control_xy = np.array([[cp[0, 0], cp[1, 0]] for cp in cp_list])

    # Full 3D control points (x, y, theta) and duration for diffusion training
    control_points_3d = np.array([cp.flatten() for cp in cp_list])  # (n_cp, 3)
    duration = traj.end_time() - traj.start_time()

    return times, S, cpts, knot_xy, control_xy, control_points_3d, duration


# ─────────────────────────────────────────────────────────────────────────
# Feedback tracking controller
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class TrackingGains:
    """PD gains for the cascaded tracking controller.

    Outer loop (position → desired acceleration):
        ax_des = ax_ff + Kp_pos*(x_ref - x) + Kd_pos*(vx_ref - vx)
        ay_des = ay_ff + Kp_pos*(y_ref - y) + Kd_pos*(vy_ref - vy)

    Inner loop (attitude → side thrust via torque):
        alpha_des = alpha_ff + Kp_att*(theta_cmd - theta) + Kd_att*(omega_ref - omega)
        Fs = alpha_des * I / torque_arm
    """
    Kp_pos: float = 4.0     # position proportional
    Kd_pos: float = 4.0     # velocity derivative
    Kp_att: float = 50.0    # attitude proportional
    Kd_att: float = 10.0    # attitude derivative


DEFAULT_GAINS = TrackingGains()

# Side engine force → action scale (not SIDE_MAX; this is the raw impulse/dt)
SIDE_FORCE_MAX = ll.SIDE_ENGINE_POWER / ll.DT


def track(plan_times, plan, gains=None, dt=None):
    """Track a planned trajectory using cascaded PD feedback control.

    Simulates through lander_step at the Box2D timestep.  At each step,
    interpolates the reference trajectory and computes corrective thrust
    commands via feedforward + PD feedback.

    Controller architecture (cascaded):

    1. **Outer loop** — position/velocity PD computes desired world-frame
       accelerations (ax_des, ay_des).

    2. **Thrust computation** — inverse dynamics maps desired accelerations
       at the *current* theta to main engine thrust Fm.

    3. **Attitude command** — desired theta is computed from the desired
       acceleration vector: theta_cmd = atan2(-ax_des, ay_des + g).
       Blended with the plan's theta_ref to avoid overcorrection.

    4. **Inner loop** — attitude PD computes desired angular acceleration,
       which maps to side thrust Fs via the torque equation.

    Parameters
    ----------
    plan_times : (n,) ndarray
        Sample times from the planner.
    plan : dict
        Plan arrays with keys: x, y, theta, vx, vy, omega,
        ax, ay, alpha, Fm, Fs.
    gains : TrackingGains, optional
    dt : float, optional
        Simulation timestep.  Defaults to ll.DT (0.02 s).

    Returns
    -------
    history : dict
        Tracked trajectory with keys:
        - t, x, y, theta, vx, vy, omega : actual state
        - x_ref, y_ref, theta_ref       : reference state
        - Fm_cmd, Fs_cmd                : commanded thrusts
        - main_action, side_action      : clipped actions sent to physics
    """
    if gains is None:
        gains = DEFAULT_GAINS
    if dt is None:
        dt = ll.DT

    p = plan
    t_end = plan_times[-1]
    n_steps = int(t_end / dt)

    # Initial state from plan [x, y, theta, vx, vy, omega]
    state = [p["x"][0], p["y"][0], p["theta"][0],
             p["vx"][0], p["vy"][0], p["omega"][0]]

    keys = ("t", "x", "y", "theta", "vx", "vy", "omega",
            "x_ref", "y_ref", "theta_ref",
            "Fm_cmd", "Fs_cmd", "main_action", "side_action")
    H = {k: np.empty(n_steps) for k in keys}

    for step in range(n_steps):
        t = step * dt
        x, y, theta, vx, vy, omega = state

        # ── Interpolate reference ───────────────────────────────────
        x_ref = float(np.interp(t, plan_times, p["x"]))
        y_ref = float(np.interp(t, plan_times, p["y"]))
        th_ref = float(np.interp(t, plan_times, p["theta"]))
        vx_ref = float(np.interp(t, plan_times, p["vx"]))
        vy_ref = float(np.interp(t, plan_times, p["vy"]))
        om_ref = float(np.interp(t, plan_times, p["omega"]))
        ax_ref = float(np.interp(t, plan_times, p["ax"]))
        ay_ref = float(np.interp(t, plan_times, p["ay"]))
        al_ref = float(np.interp(t, plan_times, p["alpha"]))

        Fm, Fs = _tracking_step(
            x, y, theta, vx, vy, omega,
            x_ref, y_ref, th_ref, vx_ref, vy_ref, om_ref,
            ax_ref, ay_ref, al_ref, gains)

        # ── Convert to actions and clip ─────────────────────────────
        main_action = float(np.clip(2.0 * Fm / THRUST_MAX - 1.0, -1.0, 1.0))
        side_action = float(np.clip(Fs / SIDE_FORCE_MAX, -1.0, 1.0))

        # ── Record ──────────────────────────────────────────────────
        H["t"][step] = t
        H["x"][step], H["y"][step], H["theta"][step] = x, y, theta
        H["vx"][step], H["vy"][step], H["omega"][step] = vx, vy, omega
        H["x_ref"][step], H["y_ref"][step], H["theta_ref"][step] = x_ref, y_ref, th_ref
        H["Fm_cmd"][step], H["Fs_cmd"][step] = Fm, Fs
        H["main_action"][step] = main_action
        H["side_action"][step] = side_action

        # ── Step physics ────────────────────────────────────────────
        state = ll.lander_step(state, Fm, Fs, dt)

    return H


def _tracking_step(x, y, theta, vx, vy, omega,
                   x_ref, y_ref, th_ref, vx_ref, vy_ref, om_ref,
                   ax_ref, ay_ref, al_ref, gains):
    """One step of the cascaded PD tracking controller.

    Returns (Fm, Fs) in force units (not actions).
    """
    import math
    ct = math.cos(theta)
    st = math.sin(theta)

    # ── Outer loop: position/velocity PD → desired acceleration ─
    ax_des = ax_ref + gains.Kp_pos * (x_ref - x) + gains.Kd_pos * (vx_ref - vx)
    ay_des = ay_ref + gains.Kp_pos * (y_ref - y) + gains.Kd_pos * (vy_ref - vy)

    # ── Main engine thrust (inverse dynamics at current theta) ──
    Fm = MASS * (-ax_des * st + (ay_des + GRAVITY) * ct)

    # ── Attitude command from desired acceleration vector ───────
    # theta_cmd points the rocket so main thrust aligns with (ax_des, ay_des+g)
    thrust_y = ay_des + GRAVITY
    thrust_x = ax_des
    theta_cmd = math.atan2(-thrust_x, thrust_y)

    # Blend plan theta with commanded theta (avoid oversteering
    # when position errors are small)
    theta_target = 0.4 * theta_cmd + 0.6 * th_ref

    # ── Inner loop: attitude PD → side thrust via torque ────────
    alpha_des = al_ref + gains.Kp_att * (theta_target - theta) + gains.Kd_att * (om_ref - omega)
    torque_arm = (2 * SIDE_AWAY * st * ct
                  + SIDE_ARM_A * st * st
                  - SIDE_ARM_B * ct * ct)
    if abs(torque_arm) > 1e-6:
        Fs = alpha_des * INERTIA / torque_arm
    else:
        Fs = 0.0

    return Fm, Fs


if __name__ == "__main__":
    times, s, *_ = solve()
    print(f"Duration     : {times[-1]:.2f} s")
    print(f"Final pos    : ({s['x'][-1]:.3f}, {s['y'][-1]:.3f})")
    print(f"Final vel    : ({s['vx'][-1]:.3f}, {s['vy'][-1]:.3f})")
    print(f"Thrust  Fm   : [{s['Fm'].min():.2f}, {s['Fm'].max():.2f}]")
    print(f"Thrust  Fs   : [{s['Fs'].min():.2f}, {s['Fs'].max():.2f}]")

    print("\n── Feedback tracking ──")
    H = track(times, s)
    pos_err = np.hypot(H["x"] - H["x_ref"], H["y"] - H["y_ref"])
    th_err = np.abs(H["theta"] - H["theta_ref"])
    print(f"Position RMS : {np.sqrt(np.mean(pos_err**2)):.4f} m")
    print(f"Position max : {pos_err.max():.4f} m")
    print(f"Theta RMS    : {np.degrees(np.sqrt(np.mean(th_err**2))):.2f} deg")
    print(f"Final pos    : ({H['x'][-1]:.3f}, {H['y'][-1]:.3f})")
    print(f"Final vel    : ({H['vx'][-1]:.3f}, {H['vy'][-1]:.3f})")
