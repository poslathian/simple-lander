"""Type stubs for solver.py — KTO trajectory optimization."""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ── Constants ────────────────────────────────────────────────────────────
SCALE: float
VIEWPORT_W: int
VIEWPORT_H: int
W: float
H: float
PAD_X: float
PAD_Y: float

GRAVITY: float
MASS: float
INERTIA: float

THRUST_MAX: float
SIDE_MAX: float

SIDE_ARM_A: float
SIDE_ARM_B: float
SIDE_AWAY: float

START: NDArray[np.float64]
GOAL: NDArray[np.float64]

SPLINE_ORDER: int

# ── Strategy ─────────────────────────────────────────────────────────────

@dataclass
class Strategy:
    name: str = ...
    warmstart_frac: float = ...
    goal_region: bool = ...
    goal_cost_weight: float = ...
    energy_cost: float = ...
    duration_cost: float = ...
    num_control_points: int = ...
    constraint_scale: float = ...

    @property
    def num_dynamics_samples(self) -> int: ...

STRATEGIES: dict[str, Strategy]

# ── Public API ───────────────────────────────────────────────────────────

def solve(
    start: ArrayLike | None = ...,
    goal: ArrayLike | None = ...,
    obstacles: Sequence[tuple[float, float, float]] = ...,
    terrain: tuple[ArrayLike, ArrayLike] | None = ...,
    on_progress: Callable[[str, float], None] | None = ...,
    max_iters: int | None = ...,
    time_budget: float = ...,
    warmstart_budget: float = ...,
    strategy: Strategy | str | None = ...,
) -> tuple[
    NDArray[np.float64],                          # times (n,)
    dict[str, NDArray[np.float64]],               # plan
    NDArray[np.float64],                          # constraint_xy (n_dyn, 2)
    NDArray[np.float64],                          # warm_xy (300, 2)
    NDArray[np.float64],                          # knot_xy (n_knots, 2)
    NDArray[np.float64],                          # control_xy (n_cp, 2)
    NDArray[np.float64],                          # control_points_3d (n_cp, 3)
    float,                                        # duration
]: ...
