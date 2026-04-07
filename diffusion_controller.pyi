"""KTODiffusionController — KTO warm-start + diffusion refinement with PD tracking.

Stateful controller that owns both the KTO guidance spline and the diffusion
model. One PD tracking loop over a blended position spline.

Conditioning: 21 dims (20 state + 1 CFG).
Model output: 30 dims (10 CPs x 3 channels: x, y, theta).
"""

from enum import IntEnum
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

# ── Coordinate types ──────────────────────────────────────────────────────

Position = tuple[float, float, float]      # (x, y, theta)
Velocity = tuple[float, float, float]      # (x', y', theta')
ThrustVec = tuple[float, float]            # (v, h) each in [-1, 1]

# ── Outcome ───────────────────────────────────────────────────────────────

class Outcome(IntEnum):
    """Ternary episode outcome for classifier-free guidance."""
    FAIL = -1
    UNKNOWN = 0
    SUCCESS = 1

# ── Input types ───────────────────────────────────────────────────────────

class ObstacleRelative(NamedTuple):
    """Nearest collision geometry relative to current observation.
    Wired to (0, 0, 0) until obstacles are implemented.
    """
    dx: float             # obstacle.x - q_now.x
    dy: float             # obstacle.y - q_now.y
    r: float              # obstacle radius

class WaypointTarget(NamedTuple):
    """Single waypoint: position and velocity error relative to current obs."""
    dq: Position          # (q_target - q_now): where to go
    dq_prime: Velocity    # (q'_target - q'_now): desired velocity delta

# ── Output types ──────────────────────────────────────────────────────────

class PositionRef(NamedTuple):
    """Position reference at a single timestep."""
    x: float
    y: float
    theta: float

# ── Constants ─────────────────────────────────────────────────────────────

DT: float             # 0.02 — simulation timestep (50 Hz)
STATE_DIM: int        # 20
CFG_DIM: int          # 1
COND_DIM: int         # 21
X_DIM: int            # 30 (10 CPs x 3 channels)
N_CPS: int            # 10
N_CHANNELS: int       # 3 (x, y, theta)

# ── DiffusionModel ────────────────────────────────────────────────────────

class ModelProtocol(Protocol):
    """Interface for the diffusion model (or NoiseModel stub)."""
    def predict(
        self,
        cond: NDArray[np.float32],
        outcome: Outcome,
        guidance_scale: float,
    ) -> NDArray[np.float64]: ...

class DiffusionModel:
    """Neural network: 21-dim conditioning -> 30-dim position CPs.

    Wraps MLP denoiser + DDIM sampler + CFG.
    Output is 10 control points x 3 (x, y, theta) in lander-relative coords.
    First CP pinned to (0,0,0), second CP constrained by q_prev for C1 continuity.
    """

    def __init__(self, checkpoint_path: str | None = ...) -> None: ...

    def predict(
        self,
        cond: NDArray[np.float32],
        outcome: Outcome,
        guidance_scale: float = ...,
    ) -> NDArray[np.float64]:
        """Run DDIM sampling with CFG.

        Args:
            cond: (20,) state conditioning vector (no CFG dim).
            outcome: desired outcome for CFG guidance.
            guidance_scale: CFG weight.

        Returns:
            (10, 3) position control points in lander-relative coords.
        """
        ...

class NoiseModel:
    """Drop-in for DiffusionModel that returns random CPs. For testing."""
    def predict(
        self,
        cond: NDArray[np.float32],
        outcome: Outcome,
        guidance_scale: float = ...,
    ) -> NDArray[np.float64]: ...

# ── KTODiffusionController ────────────────────────────────────────────────

class KTODiffusionController:
    """Stateful controller: KTO warm-start + diffusion refinement + PD tracking.

    Lifecycle:
        1. __init__: create with env, model, config
        2. warm_start(): solve KTO, store guidance spline
        3. Loop:
           a. inference() at target_frequency (3 Hz) — run diffusion model
           b. get_action() every sim step (50 Hz) — blend splines, PD track

    Spline time alignment:
        kto_spline(t_kto) sampled from t_kto_0 = t_sim at warm_start time.
        diff_spline(t_diff) sampled from t_diff_0 = t_sim at inference time.
        These have DIFFERENT time origins. Both converted to world frame before blending.
    """

    def __init__(
        self,
        env: ...,
        model: ModelProtocol | None = ...,
        target_frequency: float = ...,
        action_horizon: float = ...,
        outcome: Outcome = ...,
    ) -> None:
        """
        Args:
            env:              Gymnasium LunarLander env.
            model:            DiffusionModel or NoiseModel. None = NoiseModel.
            target_frequency: How often to call inference(), in Hz (default 3).
            action_horizon:   Duration of diffusion spline in seconds (default 1.5).
            outcome:          Desired outcome for CFG guidance.
        """
        ...

    def warm_start(
        self,
        waypoint: WaypointTarget,
        time_budget: float = ...,
    ) -> None:
        """Solve KTO to produce guidance spline.

        Stores kto_spline with t_kto_0 = current t_sim.
        After plan exhausts: get_action returns zero thrust.

        Args:
            waypoint:     Target relative to current position.
            time_budget:  Solver time budget in seconds.
        """
        ...

    def inference(self) -> None:
        """Run diffusion model at current observation.

        Builds 21-dim conditioning from class state, runs model.predict(),
        stores diff_spline with t_diff_0 = current t_sim.

        Should be called at target_frequency (default 3 Hz).
        """
        ...

    def get_action(self, guidance_margin: float = ...) -> ThrustVec:
        """Blend KTO + diffusion splines, PD track the result.

        Called every sim step (50 Hz).

        Evaluates:
          kto_ref  = kto_spline(t_sim - t_kto_0)   → world position
          diff_ref = diff_spline(t_sim - t_diff_0)  → world position
          q_ref    = (1 - margin) * kto_ref + margin * diff_ref

        PD tracks q_ref using solver._tracking_step.

        If KTO plan is exhausted (t_sim - t_kto_0 > kto_duration),
        returns zero thrust (0, 0).

        Args:
            guidance_margin: [0, 1] how far model may deviate from KTO.
                             0 = ignore model, track KTO exactly.
                             1 = ignore KTO, trust model fully.

        Returns:
            (thrust_v, thrust_h) each in [-1, 1].
        """
        ...
