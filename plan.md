# Plan: Position Diffusion Controller

DiffusionController exports just DiffusionController but defines three modules:
0. **DiffusionController** — The thing we wire up in lunar_lander.py
1. **DiffusionModel** — The actual neural network (conditioning -> position CPs)
2. **DiffusionAction** — Wraps the neural network with a PD tracking controller that enforces guidance margin

## Summary

Replace the complex 131-dim thrust-spline diffusion controller with a simplified
21-dim position-spline controller. Single waypoint, single guidance, single
ternary outcome. Output switches from thrust CPs to position CPs with PD tracking.

## Interface Changes (diffusion_controller.pyi)

### Remove
- `Obstacle`, 5-slot obstacle array
- `MaxLen` annotation class
- `WaypointResult`, `ActionResult`, `ContactGuidance`, `CrashGuidance` union types
- `CFGItem` union type
- `target_frequency` parameter
- Explicit velocity q' (implied by q_now, q_prev)
- Explicit thrust state (not needed — C2 continuity from position pairs)
- Lists-of-5 for waypoints, guidance, and CFG items

### Simplify
- `WaypointTarget` → 6 dims: delta_q (3) + delta_q' (3), relative to current obs
- `GuidanceAction` → suggested q (3 dims) at t_obs_cmd_latency
- New `Outcome` enum: `SUCCESS = 1`, `FAIL = -1`, `UNKNOWN = 0`
- New `GuidanceMargin` float in [0, 1], enforced by DiffusionAction (not in conditioning)
- Single `outcome: Outcome` replaces `classifier_free_guidance: list[CFGItem]`

### Output Change
- DiffusionModel output: `PositionSpline` (time -> (x, y, theta) relative position reference)
- DiffusionAction: PD tracking converts position refs to thrust, enforces guidance margin

## Conditioning Vector Layout (21 dims)

| Range | Dims | Field | Notes |
|-------|------|-------|-------|
| 0     | 1    | t_obs_cmd_latency | Observation-to-command delay |
| 1-3   | 3    | q_now (x, y, theta) | Current position at t0, origin of output spline |
| 4-6   | 3    | q_prev (x, y, theta) | Position at t0 - dt; with q_now implies velocity for C2 continuity |
| 7-9   | 3    | pad_rel (dx, dy, r) | Landing pad relative to q_now: (pad.x - q.x, pad.y - q.y, pad_radius) |
| 10-15 | 6    | waypoint: delta_q (3) + delta_q' (3) | Target position/velocity error relative to current obs |
| 16-18 | 3    | guidance_q (x, y, theta) | Caller's suggested position at t_obs_cmd_latency |
| 19    | 1    | action_horizon | Duration of output spline (default 1.5s) |
| —     | **STATE_DIM = 20** | | |
| 20    | 1    | outcome | CFG dim: -1/0/+1 scaled by alpha |
| —     | **COND_DIM = 21** | | |

## Model Output (30 dims)

- 10 control points x 3 channels (x, y, theta) = 30 dims
- Position B-spline in lander-relative coordinates (origin = q_now at call time)
- First CP pinned to (0, 0, 0) for C0 continuity (we are at our own origin)
- Second CP constrained by q_prev for C1 continuity (velocity matching)
- 8 free CPs for trajectory shaping

## Module Responsibilities

### DiffusionModel
- Owns the neural network (MLP denoiser) and DDIM sampler
- Input: 21-dim conditioning vector
- Output: 30-dim position CP vector
- Handles CFG: unconditional = zero the outcome dim
- Pure inference, no physics

### DiffusionAction
- Wraps DiffusionModel output into a closed-loop controller
- Evaluates PositionSpline at current time to get reference (x, y, theta)
- PD tracking controller converts position error to thrust commands
- Enforces GuidanceMargin [0, 1]: blends diffusion output with guidance_q
  - margin=0: pure diffusion (ignore guidance)
  - margin=1: hard clamp to guidance suggestion
- Returns ThrustVec (v, h) each step

### DiffusionController
- Top-level wiring: builds conditioning, calls DiffusionModel, wraps in DiffusionAction
- The single callable that lunar_lander.py imports

## Implementation Steps

1. **diffusion_controller.pyi** — Simplified interface stub (3 modules)
2. **diffusion_controller.py** — Rewrite implementation:
   - `_build_cond` produces 21-dim vector
   - `DiffusionModel` class wrapping MLP + DDIM
   - `DiffusionAction` class with PD tracking + guidance margin
   - `DiffusionController` function wiring it together
3. **model.py** — Update dimensions:
   - `COND_DIM = 21`, `STATE_DIM = 20`, `CFG_DIM = 1`
   - `X_DIM = 30` (10 CPs x 3, not 15 CPs x 2)
   - Can try hidden=256 first given 21-dim conditioning
4. **train.py** — Update dataset to produce 21-dim cond vectors
5. **eval.py** — Update to use new interface
6. **guidance_controller.py** — Simplify: outputs position guidance, not thrust

## Why This Works

- 84% reduction in conditioning dims (131 -> 21) means faster training
- Position splines + PD tracking validated in prior experiments
- q_now + q_prev gives implicit velocity without dedicating 5 conditioning dims
- GuidanceMargin is runtime, not learned — keeps model simple
- PD controller handles closed-loop stability; diffusion only plans trajectory shape
