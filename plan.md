# Plan: Position Diffusion Controller

## Summary

Replace the complex 131-dim thrust-spline diffusion controller with a simplified
32-dim position-spline controller. Single waypoint, single guidance, single
ternary outcome. Output switches from thrust CPs to position CPs with PD tracking.

## Interface Changes (diffusion_controller.pyi)

### Remove
- `Obstacle`, 5-slot obstacle array (never populated in eval)
- `MaxLen` annotation class
- `WaypointResult`, `ActionResult`, `ContactGuidance`, `CrashGuidance` union types
- `CFGItem` union type
- `target_frequency` parameter
- Lists-of-5 for waypoints, guidance, and CFG items

### Simplify
- `WaypointTarget` stays but is a single required arg (not list)
- `ActionTarget` stays but is a single required arg (not list)
- New `Outcome` enum: `SUCCESS = 1`, `FAIL = -1`, `UNKNOWN = 0`
- Single `outcome: Outcome` replaces `classifier_free_guidance: list[CFGItem]`

### Output Change
- Output type changes from `ThrustSpline` (time -> thrust) to `PositionSpline`
  (time -> (x, y, theta) relative position reference)
- Caller is responsible for PD tracking to convert position refs to thrust

## Conditioning Vector Layout (32 dims)

| Range  | Dims | Field                    |
|--------|------|--------------------------|
| 0      | 1    | t_obs_cmd_latency        |
| 1-3    | 3    | q (x, y, theta)          |
| 4-6    | 3    | q' (vx, vy, omega)       |
| 7-8    | 2    | thrust (v, h)            |
| 9-11   | 3    | contacts (l, r, body)    |
| 12-23  | 12   | waypoint (1x12)          |
| 24-29  | 6    | guidance_action (1x6)    |
| 30     | 1    | action_horizon           |
| **31** | **STATE_DIM = 31**       |
| 31     | 1    | outcome (CFG dim)        |
| **32** | **COND_DIM = 32**        |

## Model Output (30 dims)

- 10 control points x 3 channels (x, y, theta) = 30 dims
- Position B-spline in lander-relative coordinates
- PD tracking controller converts position refs to thrust commands

## Implementation Steps

1. **diffusion_controller.pyi** - Simplified interface stub
2. **diffusion_controller.py** - Rewrite implementation:
   - Simpler `_build_cond` (32 dims vs 131)
   - New `PositionSpline` output type
   - PD tracking wrapper
3. **model.py** - Update dimensions:
   - `COND_DIM = 32`, `STATE_DIM = 31`, `CFG_DIM = 1`
   - `X_DIM = 30` (stays same, but now 10x3 not 15x2)
   - Smaller model viable: hidden=256 may suffice with 32-dim cond
4. **train.py** - Update dataset to produce 32-dim cond vectors
5. **eval.py** - Update to use new interface + PD tracking
6. **guidance_controller.py** - Simplify to match new interface

## Why This Works

- 75% reduction in conditioning dims (131 -> 32) means faster training
- Position splines + PD tracking validated in prior experiments (see memory)
- Single waypoint/guidance/outcome eliminates unused zero-padding
- PD controller handles closed-loop stability, diffusion only plans trajectory
