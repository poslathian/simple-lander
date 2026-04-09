"""Clamped uniform cubic B-spline fitting for 2D position windows.

A clamped 10-CP cubic B-spline:
  degree p = 3
  n_cp   = 10
  knots  = [0,0,0,0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1,1,1,1]   # length 14

The basis is fixed, so fitting any sample window is a single linear least
squares solve `B @ cps = points`, with `cps` shape (10, 2).
"""

import numpy as np
from scipy.interpolate import BSpline

DEGREE = 3
N_CP = 10

# Open uniform clamped knot vector on [0, 1]
KNOTS = np.concatenate([
    np.zeros(DEGREE + 1),
    np.arange(1, N_CP - DEGREE) / (N_CP - DEGREE),
    np.ones(DEGREE + 1),
]).astype(np.float64)
assert KNOTS.shape == (N_CP + DEGREE + 1,)


def design_matrix(u: np.ndarray) -> np.ndarray:
    """Return B-spline basis matrix B of shape (len(u), N_CP) for parameters
    u in [0, 1]. Boundary samples u=0 and u=1 are handled correctly via the
    clamped knot vector."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    B = BSpline.design_matrix(u, KNOTS, DEGREE).toarray()
    return B  # (n, N_CP)


# Pre-computed basis for sampling N points uniformly on [0, 1]
def uniform_basis(n_samples: int) -> np.ndarray:
    return design_matrix(np.linspace(0.0, 1.0, n_samples))


def fit_window(points: np.ndarray) -> np.ndarray:
    """Fit a clamped 10-CP cubic B-spline to a uniform 2D point window.

    points: (n, 2) — assumed uniformly sampled in time across the window.
    Returns: (10, 2) control points.
    """
    n = points.shape[0]
    B = uniform_basis(n)  # (n, 10)
    # Least squares: minimise || B cps - points ||
    cps, *_ = np.linalg.lstsq(B, points, rcond=None)
    return cps  # (10, 2)


def eval_spline(cps: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate a clamped 10-CP B-spline at parameter values u in [0,1].

    cps: (10, 2) or (10,)
    u  : (n,)
    Returns: (n, 2) or (n,)
    """
    B = design_matrix(u)
    return B @ cps
