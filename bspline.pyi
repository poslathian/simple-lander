"""Clamped uniform cubic B-spline (degree 3, 10 control points).

The basis is fixed across the package: knot vector
``[0,0,0,0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1,1,1,1]`` on the unit interval.
"""

import numpy as np

DEGREE: int           # 3
N_CP: int             # 10
KNOTS: np.ndarray     # shape (14,), dtype float64


def design_matrix(u: np.ndarray) -> np.ndarray:
    """Return the (len(u), 10) basis matrix evaluated at parameters
    ``u`` in [0, 1]. Boundary samples u=0 and u=1 are handled by the clamped
    knot vector. Out-of-range u values are clipped."""


def uniform_basis(n_samples: int) -> np.ndarray:
    """Return ``design_matrix(np.linspace(0, 1, n_samples))`` —
    shape (n_samples, 10)."""


def fit_window(points: np.ndarray) -> np.ndarray:
    """Least-squares fit a clamped 10-CP cubic B-spline to a uniform 2D point
    window.

    Args:
        points: (n, 2) — uniformly sampled in time across the window.
    Returns:
        (10, 2) control points.
    """


def eval_spline(cps: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate the spline defined by ``cps`` (shape (10, 2) or (10,)) at
    parameter values ``u`` in [0, 1]. Returns (len(u), 2) or (len(u),)."""
