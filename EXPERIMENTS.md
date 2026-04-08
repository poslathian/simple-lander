# DAgger Experiment Log

## Overview

Position-diffusion DAgger: train a diffusion model to produce position B-spline
control points that replace the KTO trajectory planner at progressively higher
guidance margins. Success = matching KTO baseline landing rate at margin -> 1.0.

**Baseline model**: `position_model.pt` from the position-diffusion project.
2000 epochs on 1554 frames, hidden=256, 6 residual blocks. Starting landing
rate ~80% at margin 0.001 (pure KTO), drops to ~10% at margin 1.0 (pure model).

**KTO baseline**: ~76-82% landing rate on holdout seeds (margin 0.001).

## Common Protocol

Each round:
1. Sample N frames from archive DB (80% landed / 20% failed episodes)
2. Relabel outcome conditioning to actual episode outcome (not always +1)
3. Train for E epochs with CFG dropout (50% on outcome dim)
4. Collect 40 landed + 10 failed episodes at candidate margin (current + 0.05)
5. Evaluate on fixed holdout seeds (90000-90049)
6. Add episodes to archive DB
7. Advance margin if candidate landing rate >= 50% AND candidate margin lands >= baseline

## Experiments

### Run A — Baseline (completed)

- **Checkpoint**: `position_model_923760b.pt` (round 10), `position_model_4552a6b_r21.pt` (round 21)
- **Rounds**: 21 total (10 initial + 11 resumed)
- **Hyperparameters**:
  - frames/round: 500
  - epochs/round: 500
  - batch size: 64
  - learning rate: 1e-4
  - hidden: 256, blocks: 6
  - seed offset: 10000 (initial), 100000 (resumed)
- **Results**:
  - Margin progression: 0.200 -> 0.250 (R3) -> 0.300 (R5) -> 0.350 (R7) -> 0.400 (R10)
  - Plateaued at 0.400 for 11 rounds (R11-R21), never broke through to 0.45
  - Final eval sweep (50 seeds): m=.001: 82%, m=.2: 78%, m=.3: 80%, m=.4: 80%, m=.5: 72%, m=.7: 36%, m=1.0: 10%
  - Archive: 613 episodes, 8803 frames
- **Notes**: First run used round-varying eval seeds (not holdout). Fixed in later runs.

### Run B — 4x Data (in progress)

- **Checkpoint prefix**: `runs/B_4xdata/dagger_round*.pt`
- **Rounds**: 20
- **Hyperparameters**:
  - **frames/round: 2000** (4x baseline)
  - epochs/round: 500
  - batch size: 64
  - learning rate: 1e-4
  - hidden: 256, blocks: 6 (pre-trained weights from run A checkpoint)
  - seed offset: 200000
- **Results (in progress)**:
  - Margin progression: 0.200 -> 0.250 (R5) -> 0.300 (R8) -> 0.350 (R9) -> 0.400 (R13)
  - Reached 0.400 by round 13 (same wall as run A)
  - Holdout at round 16: m=.001: 78%, m=.40: 70%, m=.45: 60%
  - Initial landing rate: 81% (54 episodes)
- **Hypothesis**: More data per round should reduce overfitting and allow larger margins. Reached 0.4 slightly faster but hit the same wall.

### Run C — 4x Training (in progress)

- **Checkpoint prefix**: `runs/C_4xtrain/dagger_round*.pt`
- **Rounds**: 20
- **Hyperparameters**:
  - frames/round: 500
  - **epochs/round: 2000** (4x baseline)
  - **batch size: 128** (2x baseline)
  - **learning rate: 5e-5** (0.5x baseline)
  - hidden: 256, blocks: 6 (pre-trained weights from run A checkpoint)
  - seed offset: 300000
- **Results (in progress)**:
  - Margin progression: 0.200 -> 0.250 (R4) -> 0.300 (R14)
  - Slow advancement — took 10 rounds stuck at 0.250 before reaching 0.300
  - Holdout at round 18: m=.001: 76%, m=.30: 72%, m=.35: 72%
  - Loss gets very low (~0.047) but doesn't translate to better rollout performance
  - Initial landing rate: 75% (53 episodes)
- **Hypothesis**: More training epochs might extract more from each data batch. Result: likely overfitting the 500-frame sample. Lower lr + larger batch didn't help.

### Run D — Large Model h=1024 (in progress, relaunched)

- **Checkpoint prefix**: `runs/D_h1024/dagger_round*.pt`
- **Rounds**: 20
- **Hyperparameters**:
  - **frames/round: 2000** (after relaunch; originally 500)
  - **epochs/round: 1000** (after relaunch; originally 500)
  - **batch size: 128**
  - learning rate: 1e-4
  - **hidden: 1024**, blocks: 6 (**random init** — checkpoint size mismatch)
  - seed offset: 400000
- **Results (in progress)**:
  - Round 1 eval: m=.001: 74%, m=.20: 6%, m=.25: 0%
  - Model starting from scratch — loss ~0.25 (vs ~0.07 for pre-trained 256)
  - Initial collection: 0 landed in 501 episodes (model produces garbage CPs, but KTO at m=.001 still works)
- **Relaunch history**: First launched with 500 frames/500 epochs — model couldn't learn anything (2% landing). Killed and relaunched with 2000 frames/1000 epochs.
- **Hypothesis**: Larger model capacity might learn better representations. Result so far: random init is a severe handicap. Needs many rounds to catch up.

### Run E — Large Model h=2048 (in progress, relaunched)

- **Checkpoint prefix**: `runs/E_h2048/dagger_round*.pt`
- **Rounds**: 20
- **Hyperparameters**:
  - **frames/round: 2000** (after relaunch; originally 500)
  - **epochs/round: 1000** (after relaunch; originally 500)
  - **batch size: 128**
  - learning rate: 1e-4
  - **hidden: 2048**, blocks: 6 (**random init** — checkpoint size mismatch)
  - seed offset: 500000
- **Results (in progress)**:
  - Round 1 still in progress (training loss 0.34 at epoch 1000)
  - Initial collection: 1 landed in 501 episodes
- **Relaunch history**: Same as D — killed and relaunched with more data/epochs.
- **Hypothesis**: Even larger capacity. Same random-init handicap as D but worse (more parameters to learn from scratch).

## Key Observations

1. **Data volume > training intensity**: Run B (4x data) advances faster than C (4x training). Overfitting 500 frames with 2000 epochs produces low loss but poor rollout performance.

2. **Pre-trained weights matter enormously**: Runs A/B/C start at ~80% landing and advance margins. Runs D/E start from random init and can barely land at any margin > 0.001.

3. **The 0.400 wall**: Both A and B plateau at margin 0.400. At this margin the diffusion model controls 40% of the position reference — beyond this, model errors compound and cause crashes.

4. **KTO solve is the bottleneck**: ~3s per episode for the trajectory solve vs ~0.03s for rollout. KTO plan caching (modal_rollout.py) provides 6x speedup on the rollout phase.

5. **Holdout eval is essential**: Fixed seeds (90000-90049) enable cross-round comparison. Run A's first 10 rounds used varying seeds, making progress harder to track.

## File Map

- `dagger_loop.py` — Main DAgger loop (local, parametric)
- `modal_dagger.py` — Modal-scaled version with parallel rollouts + GPU training
- `modal_rollout.py` — Fast rollout collection with KTO plan caching
- `test_modal_rollout.py` — Tests for cached rollout correctness
- `position_model.pt` — Original pre-trained checkpoint (2000 epochs, 1554 frames)
- `position_model_923760b.pt` — Run A round 10 checkpoint (starting point for B/C/D/E)
- `position_model_4552a6b_r21.pt` — Run A round 21 checkpoint (best from run A)
- `runs/*/` — Per-experiment output dirs with archive.db, checkpoints, log.txt

## Reproduction

```bash
# Run A baseline
.venv/bin/python -u dagger_loop.py --checkpoint position_model.pt --rounds 20

# Run B: 4x data
.venv/bin/python -u dagger_loop.py --checkpoint position_model_923760b.pt \
  --rounds 20 --seed-offset 200000 --train-frames 2000 --run-dir runs/B_4xdata

# Run C: 4x training
.venv/bin/python -u dagger_loop.py --checkpoint position_model_923760b.pt \
  --rounds 20 --seed-offset 300000 --train-epochs 2000 --batch-size 128 --lr 5e-5 \
  --run-dir runs/C_4xtrain

# Run D: h=1024 (random init, needs more data)
.venv/bin/python -u dagger_loop.py --checkpoint position_model_923760b.pt \
  --rounds 20 --seed-offset 400000 --train-frames 2000 --train-epochs 1000 \
  --batch-size 128 --hidden 1024 --run-dir runs/D_h1024

# Run E: h=2048 (random init, needs more data)
.venv/bin/python -u dagger_loop.py --checkpoint position_model_923760b.pt \
  --rounds 20 --seed-offset 500000 --train-frames 2000 --train-epochs 1000 \
  --batch-size 128 --hidden 2048 --run-dir runs/E_h2048
```
