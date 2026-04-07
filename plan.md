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
| 7-9   | 3    | obstacle (dx, dy, r) | Nearest collision geometry relative to q_now. Wired to (0,0,0) until obstacles are implemented |
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

### Step 0: Smoke Test — Noise Model + Guidance Clamp (FIRST)

Validate the DiffusionAction PD tracking + guidance margin pipeline end-to-end
before building any neural network. This isolates the control path from the
learning path.

**What to build:**
- `DiffusionAction` with PD tracking (reuse `solver._tracking_step` gains/logic)
- `DiffusionController` wiring that accepts a `DiffusionModel`-like interface
- `NoiseModel`: drop-in replacement for `DiffusionModel` that returns random CPs
  (pure noise — the model output is irrelevant when guidance_margin ≈ 1)

**Test script: `test_noise_baseline.py`**
1. Register the LunarLander env, loop over 100 seeds
2. For each seed:
   - Create a `KTOController` (the expert)
   - Each step: extract `q_now` from Box2D state, compute `q_prev` from last step
   - Build `GuidanceAction` from KTO's plan reference position at `t + dt`
   - Call `DiffusionController` with `NoiseModel`, `guidance_margin=0.999`
   - The tight margin means DiffusionAction should hard-clamp to KTO guidance
   - Record per-step tracking error: `|q_actual - q_ref|`
3. Collect: landing rate, mean reward, per-episode position tracking RMS
4. Save frames for one seed to `./frames/` for visual inspection

**Pass criteria:**
- Landing rate >= 90% (matching KTO baseline)
- Mean position tracking RMS < 0.5m (PD controller tracks the guidance)
- Visual: trajectory follows KTO plan, smooth landing

**Why this first:**
- Validates DiffusionAction's PD controller works before any training
- Validates guidance_margin clamp logic (margin ≈ 1 → pure KTO passthrough)
- Proves the new interface wires into lunar_lander.py correctly
- If this fails, the bug is in PD/wiring, not the neural network
- Establishes the performance ceiling: this is the best DiffusionController
  can ever do (it's literally running KTO with extra steps)

### Step 1: Interface
1. **diffusion_controller.pyi** — Simplified interface stub (3 modules)

### Step 2: Implementation
2. **diffusion_controller.py** — Rewrite implementation:
   - `_build_cond` produces 21-dim vector
   - `DiffusionModel` class wrapping MLP + DDIM
   - `DiffusionAction` class with PD tracking + guidance margin
   - `DiffusionController` function wiring it together

### Step 3: Model
3. **model.py** — Update dimensions:
   - `COND_DIM = 21`, `STATE_DIM = 20`, `CFG_DIM = 1`
   - `X_DIM = 30` (10 CPs x 3, not 15 CPs x 2)
   - Can try hidden=256 first given 21-dim conditioning

### Step 4: Training + Eval
4. **train.py** — Update dataset to produce 21-dim cond vectors
5. **eval.py** — Update to use new interface
6. **guidance_controller.py** — Simplify: outputs position guidance, not thrust

## Why This Works

- 84% reduction in conditioning dims (131 -> 21) means faster training
- Position splines + PD tracking validated in prior experiments
- q_now + q_prev gives implicit velocity without dedicating 5 conditioning dims
- GuidanceMargin is runtime, not learned — keeps model simple
- PD controller handles closed-loop stability; diffusion only plans trajectory shape
