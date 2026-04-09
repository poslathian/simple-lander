# Position DAgger — Current Status

## Architecture
- DiffusionMLP (256 hidden, 6 blocks, 435K params)
- 10 CPs × 3 channels (x, y, theta) = 30-dim output
- DDIM sampler (10 steps, no CFG)
- Outcome conditioning as regular input (not classifier-free guidance)
- Observed-trajectory supervision (fit CPs to what lander actually did)

## Normalization
- x, y: normalized by 30.0 (screen width, physics-based: vx_max × action_horizon)
- theta: world radians, normalized by 4.5 (omega_max × action_horizon)
- Model output in [-1, 1], clipped
- x_mean/x_std recomputed from training data each round

## Guidance Mechanism
- Weighted average: `ref = (1-m)*kto + m*diffusion`
- NOT a clamp (clamp creates hard boundary → oscillation/crashes)
- Margin ∈ [0, 1]: 0 = pure KTO, 1 = pure diffusion

## DAgger Loop
- Start at margin 0.001
- Step ±0.001: ≥70% landing → +0.001, <70% → -0.001, floor 0.001
- Collect 40 landed episodes per round, store up to 10 failures
- Train on 500 frames sampled from archive, 200 epochs, batch 32, lr 5e-5
- KTO plan pool cached to disk (550 plans)
- Single-seed mode (--single-seed) for memorization experiments

## Evals Per Round
- Holdout landing rate at current margin (50 episodes)
- Outcome=-1 landing rate (20 episodes) — OOD detector
- Temporal consistency (10 episodes) — prediction coherence metric

## Key Findings

### Clamp vs Weighted Average
Clamp (`clip(diff, kto-m, kto+m)`) creates hard boundary that causes oscillation.
At small margins, untrained model CPs saturate at clamp boundary every step.
Weighted average is smooth and proportional — model can't hurt at low margins.
See issues.md for full writeup.

### Outcome Conditioning as OOD Detector
When outcome=+1 lands more than outcome=-1: model is in-distribution.
When gap inverts: model is OOD, shouldn't be trusted at current margin.
Works best at higher margins where model has more influence.

### Temporal Consistency
Compare overlapping 666ms windows between successive inference predictions.
Untrained: ~35 world units. Trained (R67): **1.08 world units** (median 0.68).
Tracks learning progress even when margin stalls.

### Single-Seed Memorization
Training on one seed is 3-10x faster than multi-seed at advancing margin.
The model memorizes the KTO trajectory for one initial condition, then
gradually takes on more responsibility via the weighted average blend.

## Current Experiment Results

### Single-seed, 256-hidden, strong training (200ep/500fr/batch32/lr5e-5)

**Round 67 of 100, seed=50042:**
- Margin: **0.051** (5.1% model influence)
- Consistency: **1.08** world units (median 0.68)
- 24 consecutive advances (no retreat since R43)
- Landing rate: 74-95% at every margin level
- Archive: 62k frames

**Progression:**
| Phase | Rounds | Margin range | Consistency |
|-------|--------|-------------|-------------|
| Bootstrap | R1-12 | 0.001-0.010 | 27→10 |
| First wall | R12-15 | 0.010-0.015 | 10→6 |
| Steady climb | R15-42 | 0.015-0.030 | 6→2 |
| Acceleration | R43-67 | 0.030-0.051 | 2→1.1 |

**Walls encountered and broken:**
- m=0.010 (R12): broke through R14 (2 rounds)
- m=0.015 (R19): broke through R23 (4 rounds)
- m=0.025 (R35): broke through R37 (2 rounds)
- m=0.030 (R42): broke through R44 (2 rounds)
- No wall since R42 — 24 consecutive advances

### Comparison: weak vs strong training (same 256-hidden model, single seed)

| Metric | Weak (50ep/200fr) | Strong (200ep/500fr) |
|--------|-------------------|----------------------|
| Rounds to m=0.010 | 38 | **12** |
| Rounds to m=0.020 | 78 | **30** |
| Peak margin (100 rounds) | 0.022 | **0.051+** (still climbing) |
| Consistency at peak | 7.7 | **1.08** |
| Retreats in 100 rounds | ~15 | ~5 |

### Multi-seed experiment (for comparison)
- 256-hidden, 550 seeds, strong training
- Stuck at m=0.003-0.004 after 57 rounds
- The model can't generalize across seeds at this size

## Next Steps
1. Scale to 1024-hidden on Modal (6.5M params, GPU training)
2. Use 550+ seeds for generalization
3. Fix Modal rollout code (currently crashes with ConflictError)
4. Refactor: supervision CPs from controller (TODO.md)
5. Fix t_obs_cmd_latency (should be compute time, not DT)
