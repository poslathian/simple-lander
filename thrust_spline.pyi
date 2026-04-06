"""Clean B-spline interface for fitting thrust trajectories from rollout windows.

Design:
  - Clamped cubic B-spline: first/last (degree+1) knots pinned to endpoints.
  - 15 control points, cubic (degree 3) → 19 knots.
  - First and last CPs pin the spline value at the endpoints.
  - 13 free CPs for shaping, 11 free interior knots.

Usage:
    window = RolloutWindow.from_arrays(times, actions, t_start=2.0, horizon=4.0)
    spline = ThrustSpline.fit(window)
    thrust_v, thrust_h = spline(2.5)
    cps = spline.control_points       # (15, 2) array
"""

from typing import NamedTuple, overload
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CPS: int          # 15  — number of control points per channel
DEGREE: int         # 3   — cubic
N_KNOTS: int        # 19  — N_CPS + DEGREE + 1
N_INTERIOR_SPANS: int  # 12 — N_CPS - DEGREE
SIM_DT: float       # 0.02 — 50 Hz sim timestep

# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------

class ThrustVec(NamedTuple):
    """Two-channel thrust output, each in [-1, 1]."""
    v: float   # vertical (main engine)
    h: float   # horizontal (side thrusters)

# ---------------------------------------------------------------------------
# RolloutWindow
# ---------------------------------------------------------------------------

class RolloutWindow:
    """A slice of rollout data with context for B-spline fitting.

    Holds timestamps and thrust values covering the requested window plus
    synthetic context extending into the head/tail knot regions.

    Context synthesis rules:
        Head (t < t_start): sample-and-hold from the earliest real value.
        Tail (t > t_end):   zero thrust (engine off after episode).
    """

    t_start: float
    """Window start time (action horizon begins here)."""

    t_end: float
    """Window end time."""

    times: NDArray[np.float64]
    """(M,) timestamps covering window + context. May extend beyond [t_start, t_end]."""

    thrust_v: NDArray[np.float32]
    """(M,) vertical thrust values aligned with times."""

    thrust_h: NDArray[np.float32]
    """(M,) horizontal thrust values aligned with times."""

    @property
    def horizon(self) -> float:
        """Window duration in seconds: t_end - t_start."""
        ...

    @property
    def context_before(self) -> float:
        """Seconds of data available before t_start."""
        ...

    @property
    def context_after(self) -> float:
        """Seconds of data available after t_end."""
        ...

    @classmethod
    def from_arrays(
        cls,
        times: NDArray[np.float64],
        actions: NDArray[np.float32],
        t_start: float,
        horizon: float,
        context: float | None = None,
    ) -> RolloutWindow:
        """Extract a window from raw trajectory arrays.

        Automatically synthesizes head/tail context when the rollout
        doesn't extend far enough to cover the knot range.

        Args:
            times:    (N,) monotonic timestamps at target_frequency.
            actions:  (N, 2) thrust values [v, h], each in [-1, 1].
            t_start:  desired window start time.
            horizon:  window duration in seconds (= action_horizon).
            context:  override context duration. Default: auto from knot geometry.
        """
        ...

    @classmethod
    def from_db(
        cls,
        rollout_id: int,
        t_start: float = 0.0,
        horizon: float = 4.0,
        db_path: str = "rollouts.db",
        context: float | None = None,
    ) -> RolloutWindow:
        """Load a window from a rollouts.db SQLite database.

        Reads the action_trajectory blob, reconstructs timestamps at SIM_DT,
        and delegates to from_arrays.

        Raises:
            ValueError: if rollout_id is not found.
        """
        ...

# ---------------------------------------------------------------------------
# ThrustSpline
# ---------------------------------------------------------------------------

class ThrustSpline:
    """Callable clamped cubic B-spline mapping time -> (thrust_v, thrust_h).

    Clamped knots: first/last 4 knots pinned to endpoints.
    Valid evaluation domain is [t_start, t_end].

    Knot geometry (15 CPs, degree 3, horizon H):
        19 knots total, 11 free interior knots
        knots[0:4]  = t_start  (clamped)
        knots[4:16] = uniform interior
        knots[15:19] = t_end   (clamped)
        13 free CPs for shaping, 2 pinned to endpoints
    """

    t_start: float
    """Start of valid evaluation domain."""

    t_end: float
    """End of valid evaluation domain."""

    knots: NDArray[np.float64]
    """(14,) uniform knot vector."""

    control_points: NDArray[np.float64]
    """(10, 2) control points — columns are [v, h]. The optimizable parameters."""

    @overload
    def __call__(self, t: float) -> ThrustVec: ...
    @overload
    def __call__(self, t: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def __call__(self, t: float | NDArray[np.float64]) -> ThrustVec | NDArray[np.float64]:
        """Evaluate the spline at time t.

        For scalar t: returns ThrustVec(v, h).
        For array  t: returns (N, 2) array of [v, h] pairs.
        """
        ...

    @property
    def horizon(self) -> float:
        """Action horizon in seconds: t_end - t_start."""
        ...

    @property
    def knot_spacing(self) -> float:
        """Uniform knot interval h = horizon / 7, in seconds."""
        ...

    @classmethod
    def fit(cls, window: RolloutWindow, n_cps: int = ...) -> ThrustSpline:
        """Fit an unclamped B-spline to a rollout window via least squares.

        Uses full-range collocation so that context data outside the valid
        domain constrains edge CPs.

        Args:
            window: rollout window with context (from RolloutWindow.from_arrays).
            n_cps:  number of control points (default 10).
        """
        ...

    @classmethod
    def from_control_points(
        cls,
        cps: NDArray[np.float64],
        t_start: float,
        t_end: float,
    ) -> ThrustSpline:
        """Reconstruct a spline from control points and time bounds.

        The knot vector is deterministically computed from (t_start, t_end).
        Use this after MPC optimization to get a callable spline back.

        Args:
            cps:     (10, 2) control points [v, h].
            t_start: valid domain start.
            t_end:   valid domain end.
        """
        ...

    def residuals(self, times: NDArray[np.float64], targets: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute (N, 2) prediction residuals: targets - spline(times).

        Args:
            times:   (N,) evaluation times.
            targets: (N, 2) ground truth [v, h].
        """
        ...

    def mse(self, window: RolloutWindow) -> float:
        """Mean squared error over the window domain [t_start, t_end].

        Only evaluates on data points within the valid domain, ignoring
        the synthetic context regions.
        """
        ...
