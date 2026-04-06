"""Clean B-spline interface for fitting thrust trajectories from rollout windows.

Design:
  - Clamped cubic B-spline: first/last (degree+1) knots pinned to endpoints.
  - 15 control points, cubic (degree 3) → 19 knots.
  - First and last CPs pin the spline value at the endpoints.
  - 13 free CPs for shaping, 11 free interior knots.

Usage:
    window = RolloutWindow.from_arrays(times, actions, t_start=2.0, horizon=4.0)
    spline = ThrustSpline.fit(window)
    thrust_v, thrust_h = spline(2.5)  # evaluate at any t in [t_start, t_start+horizon]
    cps = spline.control_points       # (10, 2) array for MPC optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.interpolate import BSpline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CPS = 15
DEGREE = 3
N_KNOTS = N_CPS + DEGREE + 1  # 19
N_INTERIOR_SPANS = N_CPS - DEGREE  # 12
SIM_DT = 0.02


# ---------------------------------------------------------------------------
# Rollout window
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RolloutWindow:
    """A slice of rollout data with optional context before/after the window."""
    t_start: float              # window start (action horizon begins here)
    t_end: float                # window end
    times: np.ndarray           # all timestamps (may extend beyond window)
    thrust_v: np.ndarray        # vertical thrust values
    thrust_h: np.ndarray        # horizontal thrust values

    @property
    def horizon(self) -> float:
        return self.t_end - self.t_start

    @property
    def context_before(self) -> float:
        """How much data exists before window start."""
        return max(0.0, self.t_start - self.times[0])

    @property
    def context_after(self) -> float:
        """How much data exists after window end."""
        return max(0.0, self.times[-1] - self.t_end)

    @classmethod
    def from_arrays(
        cls,
        times: np.ndarray,
        actions: np.ndarray,
        t_start: float,
        horizon: float,
        context: float | None = None,
    ) -> RolloutWindow:
        """Extract a window from raw arrays.

        Args:
            times: (N,) timestamps
            actions: (N, 2) thrust values [v, h]
            t_start: desired window start
            horizon: window duration
            context: how much surrounding data to include (default: auto)
        """
        t_end = t_start + horizon

        # With clamped knots the valid domain is exactly [t_start, t_end]
        mask = (times >= t_start - 1e-10) & (times <= t_end + 1e-10)
        t_sel = times[mask]
        v_sel = actions[mask, 0]
        h_sel = actions[mask, 1]

        return cls(
            t_start=t_start,
            t_end=t_end,
            times=t_sel,
            thrust_v=v_sel,
            thrust_h=h_sel,
        )

    @classmethod
    def from_db(
        cls,
        rollout_id: int,
        t_start: float = 0.0,
        horizon: float = 4.0,
        db_path: str = "rollouts.db",
        context: float | None = None,
    ) -> RolloutWindow:
        """Load from rollouts.db."""
        import sqlite3
        import json

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM rollouts WHERE id=?", (rollout_id,)).fetchone()
        conn.close()

        if row is None:
            raise ValueError(f"Rollout {rollout_id} not found")

        act_shape = json.loads(row["action_shape"])
        actions = np.frombuffer(row["action_trajectory"], dtype=np.float32).reshape(act_shape)
        times = np.arange(len(actions)) * SIM_DT

        return cls.from_arrays(times, actions, t_start, horizon, context)


# ---------------------------------------------------------------------------
# Thrust spline
# ---------------------------------------------------------------------------

class ThrustVec(NamedTuple):
    v: float
    h: float


@dataclass(frozen=True)
class ThrustSpline:
    """Callable B-spline over a time window, returning (thrust_v, thrust_h).

    The spline is unclamped with uniform knots. The valid evaluation domain
    is [t_start, t_end] — the action window. Evaluating outside this range
    extrapolates (use with caution).
    """
    t_start: float
    t_end: float
    knots: np.ndarray           # (14,) uniform knot vector
    control_points: np.ndarray  # (10, 2) — the optimizable parameters
    _spl_v: BSpline
    _spl_h: BSpline

    def __call__(self, t: float | np.ndarray) -> ThrustVec | np.ndarray:
        """Evaluate at time t. Returns ThrustVec for scalar, (N,2) for array."""
        v = self._spl_v(t)
        h = self._spl_h(t)
        if np.isscalar(t):
            return ThrustVec(float(v), float(h))
        return np.column_stack([v, h])

    @property
    def horizon(self) -> float:
        return self.t_end - self.t_start

    @property
    def knot_spacing(self) -> float:
        return self.horizon / N_INTERIOR_SPANS

    @classmethod
    def fit(cls, window: RolloutWindow, n_cps: int = N_CPS) -> ThrustSpline:
        """Fit unclamped B-spline to a rollout window."""
        knots = _clamped_knots(window.t_start, window.t_end)

        spl_v = _fit_channel(window.times, window.thrust_v, knots)
        spl_h = _fit_channel(window.times, window.thrust_h, knots)

        cps = np.column_stack([spl_v.c, spl_h.c])

        return cls(
            t_start=window.t_start,
            t_end=window.t_end,
            knots=knots,
            control_points=cps,
            _spl_v=spl_v,
            _spl_h=spl_h,
        )

    @classmethod
    def from_control_points(
        cls,
        cps: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> ThrustSpline:
        """Reconstruct from control points (e.g., after MPC optimization)."""
        knots = _clamped_knots(t_start, t_end)
        spl_v = BSpline(knots, cps[:, 0], DEGREE, extrapolate=True)
        spl_h = BSpline(knots, cps[:, 1], DEGREE, extrapolate=True)
        return cls(
            t_start=t_start,
            t_end=t_end,
            knots=knots,
            control_points=cps,
            _spl_v=spl_v,
            _spl_h=spl_h,
        )

    def residuals(self, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute (N, 2) residuals for fitting diagnostics."""
        pred = self(times)
        return targets - pred

    def mse(self, window: RolloutWindow) -> float:
        """MSE over the window domain."""
        mask = (window.times >= self.t_start) & (window.times <= self.t_end)
        t = window.times[mask]
        targets = np.column_stack([window.thrust_v[mask], window.thrust_h[mask]])
        res = self.residuals(t, targets)
        return float(np.mean(res ** 2))


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _clamped_knots(t_start: float, t_end: float) -> np.ndarray:
    """Clamped uniform knot vector for cubic B-spline with N_CPS control points.

    First (DEGREE+1) knots = t_start, last (DEGREE+1) knots = t_end,
    interior knots uniformly spaced. This pins the spline value at both
    endpoints to the first/last control point respectively.

    With 10 CPs, degree 3: 14 knots total, 6 interior knots free,
    2 CPs pinned to endpoints, 8 CPs free for shaping.
    """
    interior = np.linspace(t_start, t_end, N_INTERIOR_SPANS + 1)
    return np.concatenate([
        np.full(DEGREE, t_start),
        interior,
        np.full(DEGREE, t_end),
    ])


def _fit_channel(times: np.ndarray, values: np.ndarray, knots: np.ndarray) -> BSpline:
    """LSQ fit one channel using clamped knots.

    Builds the collocation matrix manually for robustness.
    Valid domain is [knots[DEGREE], knots[-DEGREE-1]] = [t_start, t_end].
    """
    t_lo, t_hi = float(knots[DEGREE]), float(knots[-DEGREE - 1])
    eps = 1e-10
    mask = (times >= t_lo - eps) & (times <= t_hi + eps)
    t_fit = times[mask]
    v_fit = values[mask]

    if len(t_fit) < N_CPS:
        t_dense = np.linspace(t_lo + eps, t_hi - eps, N_CPS * 2)
        v_fit = np.interp(t_dense, t_fit, v_fit) if len(t_fit) >= 2 else np.zeros_like(t_dense)
        t_fit = t_dense

    # Build collocation matrix: B[i, j] = N_j(t_fit[i])
    # Each basis function N_j has support [knots[j], knots[j+degree+1]].
    # We evaluate it directly via BSpline.basis_element on its local knots.
    n = len(t_fit)
    B = np.zeros((n, N_CPS))
    for j in range(N_CPS):
        local_knots = knots[j : j + DEGREE + 2]
        basis = BSpline.basis_element(local_knots, extrapolate=False)
        vals = basis(t_fit)
        vals = np.nan_to_num(vals, nan=0.0)
        B[:, j] = vals

    # Solve B @ cps = v_fit via least squares
    cps, _, _, _ = np.linalg.lstsq(B, v_fit, rcond=None)

    return BSpline(knots, cps, DEGREE, extrapolate=True)
