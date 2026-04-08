# DAgger Experiment Log

## Overview

Position-diffusion DAgger: train a diffusion model to produce position B-spline
control points that replace the KTO trajectory planner at progressively higher
guidance margins. Success = matching KTO baseline landing rate at margin -> 1.0.

**Baseline model**: `position_model.pt` from the position-diffusion project.
2000 epochs on 1554 frames, hidden=256, 6 residual blocks. Starting landing
rate ~80% at margin 0.001 (pure KTO), drops to ~10% at margin 1.0 (pure model).

**KTO baseline**: ~76-88% landing rate on holdout seeds (margin 0.001).

## Final Results

| Run | Hidden | Params | Rounds | Final Margin | m=.5 | m=.7 | m=1.0 |
|-----|--------|--------|--------|-------------|------|------|-------|
| A (baseline) | 256 | 436K | 21 | 0.400 | 72% | 36% | 10% |
| B (4x data) | 256 | 436K | 20 | 0.400 | 66% | 58% | 42% |
| C (4x train) | 256 | 436K | 20 | 0.300 | 68% | 54% | 36% |
| **D (h=1024)** | 1024 | 6.5M | 35 | **1.000** | **84%** | **82%** | **74%** |
| **E (h=2048)** | 2048 | 25.5M | 26 | **1.000** | **84%** | **76%** | **76%** |

**Winner: E (h=2048) at 76% margin=1.0**, though D (h=1024) achieved similar results
with 4x fewer parameters and reached 1.0 sooner (round 22 vs round 23).

## Common Protocol

Each round:
1. Sample N frames from archive DB (80% landed / 20% failed episodes)
2. Relabel outcome conditioning to actual episode outcome (not always +1)
3. Train for E epochs with CFG dropout (50% on outcome dim)
4. Collect 40-80 landed + 10-20 failed episodes at candidate margin (current + 0.05)
5. Evaluate on fixed holdout seeds (90000-90049)
6. Add episodes to archive DB
7. Advance margin if candidate landing rate >= 50% AND candidate meets threshold

### Advancement Rule Evolution

- **Runs A/B/C** (strict): candidate lands >= baseline lands (100% match)
- **Runs D/E** (relaxed): candidate lands >= 70% of baseline lands

The relaxed rule was critical for breaking through margin walls. D was stuck at
0.250 for 8 rounds under the strict rule, then advanced 5 times in 5 rounds
after switching to 70%.

## Experiments

### Run A — Baseline (completed)

- **Checkpoint**: `position_model_923760b.pt` (round 10), `position_model_4552a6b_r21.pt` (round 21)
- **Rounds**: 21 total (10 initial + 11 resumed)
- **Hyperparameters**:
  - frames/round: 500, epochs/round: 500, batch size: 64, lr: 1e-4
  - hidden: 256, blocks: 6, pre-trained weights
  - Local CPU training
- **Results**:
  - Margin progression: 0.200 -> 0.250 (R3) -> 0.300 (R5) -> 0.350 (R7) -> 0.400 (R10)
  - Plateaued at 0.400 for 11 rounds (R11-R21)
  - Final eval: m=.001: 82%, m=.3: 80%, m=.4: 80%, m=.5: 72%, m=.7: 36%, m=1.0: 10%
  - Archive: 613 episodes, 8803 frames
- **Notes**: First run used round-varying eval seeds. Fixed in later runs.

### Run B — 4x Data (completed)

- **Checkpoint**: `runs/B_4xdata/dagger_round20.pt`
- **Rounds**: 20
- **Hyperparameters**:
  - **frames/round: 2000** (4x baseline)
  - epochs/round: 500, batch size: 64, lr: 1e-4
  - hidden: 256, blocks: 6, pre-trained weights
  - Local CPU training
- **Results**:
  - Margin progression: 0.200 -> 0.250 (R5) -> 0.300 (R8) -> 0.350 (R9) -> 0.400 (R13)
  - Same 0.400 wall as run A
  - Final eval: m=.001: 78%, m=.4: 80%, m=.5: 66%, m=.7: 58%, m=1.0: 42%
  - Best 256-dim model at m=1.0
- **Finding**: More data per round improved high-margin performance (42% vs 10% at m=1.0)
  but couldn't break through 0.400 margin advancement.

### Run C — 4x Training (completed)

- **Checkpoint**: `runs/C_4xtrain/dagger_round20.pt`
- **Rounds**: 20
- **Hyperparameters**:
  - frames/round: 500
  - **epochs/round: 2000** (4x), **batch size: 128** (2x), **lr: 5e-5** (0.5x)
  - hidden: 256, blocks: 6, pre-trained weights
  - Local CPU training
- **Results**:
  - Margin progression: 0.200 -> 0.250 (R4) -> 0.300 (R14)
  - Stuck at 0.250 for 10 rounds before reaching 0.300
  - Final eval: m=.001: 76%, m=.5: 68%, m=.7: 54%, m=1.0: 36%
  - Loss converged very low (~0.047) but didn't translate to rollout improvement
- **Finding**: More training epochs on small data overfits. Data volume >> training intensity.

### Run D — Large Model h=1024 (completed, best efficiency)

- **Checkpoint**: `runs/D_h1024/dagger_round19.pt`
- **Rounds**: 35 total (6 CPU + 9 GPU old-rule + 20 GPU new-rule)
- **Hyperparameters**:
  - frames/round: 2000, epochs/round: 1000, batch size: 128, lr: 1e-4
  - **hidden: 1024**, blocks: 6, **random init** (size mismatch with 256-dim checkpoint)
  - GPU training on Modal T4 (~120s/round training)
  - KTO plan pool (500 seeds pre-solved, reused across rounds)
- **Results**:
  - Started from 0% landing at any margin > 0.001 (random init)
  - By round 6 (CPU): m=.20: 72%, m=.25: 66% — approaching advancement
  - Switched to GPU + 70% rule at round 10: immediately advanced
  - Margin progression: 0.250 (R10) -> 0.300 (R11) -> ... -> 1.000 (R22)
  - Final eval: m=.001: 88%, m=.5: 84%, m=.7: 82%, m=1.0: 74%
  - Archive: ~5000 episodes, ~40000 frames
- **Finding**: 6.5M params was the sweet spot — enough capacity to learn complex
  trajectories, fast enough to train in ~2 min on T4. Best params-to-performance ratio.

### Run E — Large Model h=2048 (completed, best absolute)

- **Checkpoint**: `runs/E_h2048/dagger_round20.pt` (GPU log2 round 20)
- **Rounds**: 26 total (3 CPU + 4 GPU old-rule + 1 timeout recovery + 18 GPU new-rule)
- **Hyperparameters**:
  - frames/round: 2000, epochs/round: 1000, batch size: 128, lr: 1e-4
  - **hidden: 2048**, blocks: 6, **random init**
  - GPU training on Modal T4 (~250s/round training, 97MB checkpoint)
  - KTO plan pool (500 seeds pre-solved)
- **Results**:
  - Slower start than D due to 4x more parameters
  - Hit training timeout (120s) on first GPU round — fixed to 1800s
  - Margin progression: 0.250 (R8) -> 0.300 (R9) -> ... -> 1.000 (R23)
  - Final eval: m=.001: 86%, m=.3: 86%, m=.5: 84%, m=.7: 76%, m=1.0: 76%
  - Flattest performance curve — barely degrades up to m=.5
- **Finding**: 25.5M params achieved highest absolute m=1.0 performance (76%)
  but the marginal gain over D (74%) doesn't justify the 4x training cost and
  97MB checkpoint size. The flatter curve (86% through m=.3) suggests the extra
  capacity helps maintain performance at moderate margins.

## Key Findings

### What Worked

1. **Model capacity is the primary bottleneck**: 256-dim models (436K params) plateau
   at margin 0.400 regardless of data or training. 1024-dim (6.5M) and 2048-dim (25.5M)
   both reached margin 1.000.

2. **Relaxed advancement rule (70%)**: The strict 100%-of-baseline rule caused models
   to stall for many rounds at margins where they were performing well but not perfectly.
   Switching to 70% unlocked rapid advancement without degrading final performance.

3. **GPU training**: 7x speedup on training (2 min vs 15 min per round for 1024-dim).
   Essential for making large model experiments feasible.

4. **KTO plan caching**: Pre-solving a pool of 500 KTO plans (one-time ~5 min cost)
   eliminated the 75s/round KTO solve overhead. Plans are deterministic per seed.

5. **Data volume > training intensity**: 2000 frames/round consistently outperformed
   500 frames with 4x more epochs. The model needs diverse on-policy data, not
   more gradient steps on stale data.

### What Didn't Work

1. **Outcome conditioning (CFG)**: Sanity check at margin 1.0 showed zero differentiation
   between outcome=+1, 0, and -1. The model learned to land through DAgger data
   aggregation, not through CFG-guided steering. The improvement came from seeing more
   diverse trajectories at higher margins, not from the outcome signal.

2. **Pre-trained weight transfer to larger models**: The 256-dim checkpoint can't
   initialize 1024 or 2048-dim models (size mismatch). Both started from random init,
   requiring ~6-8 rounds just to reach baseline performance. A distillation or
   progressive growing approach might help.

3. **256-dim model capacity**: Despite being well-trained (2000 epochs, 1554 frames),
   the 436K param model fundamentally can't represent trajectories well enough for
   margin > 0.400. The position B-spline space (10 CPs x 3 channels) is too complex
   for 6 layers of 256-dim residual blocks.

### Infrastructure Lessons

- **Modal app entanglement**: Importing a module that defines Modal functions registers
  them globally. Training and rollout functions must be in separate files/apps to avoid
  triggering unnecessary image builds (drake+Box2D image takes >5 min).
- **Python version matching**: Modal's `serialized=True` functions require local Python
  version to match the image. Mismatch causes silent hangs, not errors.
- **Checkpoint upload**: 97MB checkpoint (h=2048) uploads in ~7s to Modal — not a
  bottleneck, but larger models would benefit from Modal Volumes.
- **Heartbeat timeouts**: Large `starmap` calls (>100 items) can exceed Modal's heartbeat
  timeout. Chunking into batches of 100 with separate `app.run()` contexts fixes this.

## File Map

- `dagger_loop.py` — Main DAgger loop (local, parametric)
- `modal_dagger_v2.py` — Modal orchestrator: GPU training + cached parallel rollouts
- `modal_train_gpu.py` — Standalone GPU training on Modal T4
- `modal_rollout.py` — Fast rollout collection with KTO plan caching
- `test_modal_rollout.py` — Tests for cached rollout correctness
- `eval_outcome_cond.py` — Outcome conditioning sanity check
- `DAGGER_PROCEDURE.md` — Full procedure writeup
- `position_model.pt` — Original pre-trained checkpoint (2000 epochs, 1554 frames)
- `position_model_923760b.pt` — Run A round 10 checkpoint (starting point for B/C/D/E)
- `runs/*/` — Per-experiment output dirs with archive.db, checkpoints, log*.txt

## Reproduction

```bash
# Run A baseline (local CPU)
.venv/bin/python -u dagger_loop.py --checkpoint position_model.pt --rounds 20

# Run B: 4x data (local CPU)
.venv/bin/python -u dagger_loop.py --checkpoint position_model_923760b.pt \
  --rounds 20 --seed-offset 200000 --train-frames 2000 --run-dir runs/B_4xdata

# Run C: 4x training (local CPU)
.venv/bin/python -u dagger_loop.py --checkpoint position_model_923760b.pt \
  --rounds 20 --seed-offset 300000 --train-epochs 2000 --batch-size 128 --lr 5e-5 \
  --run-dir runs/C_4xtrain

# Run D: h=1024 on Modal GPU (best efficiency)
.venv/bin/python -u modal_dagger_v2.py --checkpoint position_model_923760b.pt \
  --rounds 30 --seed-offset 700000 --train-frames 2000 --train-epochs 1000 \
  --batch-size 128 --hidden 1024 --run-dir runs/D_h1024

# Run E: h=2048 on Modal GPU (best absolute)
.venv/bin/python -u modal_dagger_v2.py --checkpoint position_model_923760b.pt \
  --rounds 30 --seed-offset 800000 --train-frames 2000 --train-epochs 1000 \
  --batch-size 128 --hidden 2048 --run-dir runs/E_h2048
```
