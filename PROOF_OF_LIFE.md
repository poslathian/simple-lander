# Proof of Life: Diffusion Controller for Lunar Lander

## Summary

A diffusion model trained on 100 KTO controller demonstrations learns to predict
thrust B-spline control points, achieving 80% landing rate at guidance margins
0.1-0.2 (vs 70% baseline with tight 0.001 clamping). This confirms the model
has learned meaningful thrust patterns from the KTO teacher.

## Architecture

### Pipeline

```
KTOController.step(env)  -->  GuidanceController  -->  ActionTarget(margin)
                                                            |
                                                            v
                                                  DiffusionController
                                                    |            |
                                              model inference    post-inference
                                              (DDIM sampling)    guidance clamp
                                                    |            |
                                                    v            v
                                               ThrustSpline(DT) --> env.step()
```

### DiffusionMLP
- **Input**: noisy x (30-dim), conditioning (131-dim), timestep
- **Output**: predicted noise epsilon (30-dim)
- **Architecture**: 512-hidden, 6 residual blocks with LayerNorm + SiLU
- **Timestep embedding**: sinusoidal (64-dim) -> linear -> SiLU -> 512-dim

### Thrust Spline
- **15 control points** x 2 channels (v, h) = 30-dim output
- **Clamped cubic B-spline**: 19 knots, first/last 4 pinned to endpoints
- **13 free CPs** for shaping, 2 pinned to endpoints
- First CP pinned to observed thrust (spline continuity)

### Conditioning Vector (131-dim)
- State (119 dims): latency, position, velocity, thrust, contacts,
  5 obstacles, 5 waypoints (12 each), 5 guidance actions (6 each),
  action_horizon, target_frequency
- CFG (12 dims): waypoint results (5), action results (5), contact (1), crash (1)

### Diffusion Process
- **Schedule**: cosine, T=100
- **Sampler**: DDIM, 10 steps, deterministic (eta=0)
- **CFG**: 50% dropout during training. At inference, builds unconditional
  branch by zeroing CFG dims, blends with guidance_scale=2.0

## Training

### Data Collection
```bash
.venv/bin/python lunar_lander.py --collect --seed 0 --db rollouts.db
```

- **Controller**: KTOController (Drake trajectory optimization + PD tracking)
- **100 episodes** stored (81 landed, 14 crash, 2 flyaway, 3 timeout)
- **2276 training frames** (model-call conditioning + fitted B-spline targets)
- **Guidance margin**: random per episode, N(0.2, 0.1) clamped [0.01, 1.0]
- **Action horizon**: 3.0s fixed
- **Target frequency**: 5 Hz (model called every 10 env steps)
- Landed episodes padded with 3s of zero-thrust
- Failed episodes < 3s discarded

### Training
```bash
.venv/bin/python train.py --db rollouts.db --epochs 100 --save model.pt
```

- **Loss**: L2 noise prediction (epsilon-matching)
- **Optimizer**: AdamW, lr=1e-3
- **Batch size**: 64
- **CFG dropout**: 50% (zero CFG dims randomly per frame)
- **Normalization**: per-CP-dim mean/std computed from training targets
- **Loss curve**: 0.94 -> 0.30 over 100 epochs

### Evaluation
```bash
.venv/bin/python eval.py --model model.pt --n-baseline 20 --n-per-margin 5
```

Results:
```
  Margin     N   Land%    Reward
--------------------------------
   0.001    20    70%     -6.98    <- baseline (tight KTO clamp)
   0.100     5    80%     -5.88    <- model contributing positively
   0.200     5    80%     -6.00    <- model contributing positively
   0.300     5    40%     -7.85    <- degrading
   0.400     5    20%     -8.80
   0.500+    5     0%    -10.00    <- model alone (untrained regime)
```

The sweet spot at margin 0.1-0.2 shows the model improves over pure KTO
clamping by smoothing thrust commands within a tight band. Beyond 0.3,
the model doesn't have enough training signal to guide autonomously.

## How to Run

### Prerequisites
```bash
cd simple_lander
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Collect + Train + Eval (full pipeline)
```bash
# 1. Collect KTO demonstrations
.venv/bin/python lunar_lander.py --collect --seed 0 --db rollouts.db

# 2. Train diffusion model
.venv/bin/python train.py --db rollouts.db --epochs 100 --save model.pt

# 3. Evaluate across guidance margins
.venv/bin/python eval.py --model model.pt

# 4. Watch it fly (with trained model + tight guidance)
.venv/bin/python lunar_lander.py --diffusion --episodes 5
```

### Interactive demo
```bash
# KTO controller with keyboard override
.venv/bin/python lunar_lander.py --diffusion --keyboard

# Pure KTO (no diffusion)
.venv/bin/python lunar_lander.py --kto --episodes 5
```

## Files
| File | Purpose |
|------|---------|
| `lunar_lander.py` | Gymnasium env, KTOController, heuristic, --collect mode |
| `diffusion_controller.py` | DiffusionController inference, conditioning builder |
| `guidance_controller.py` | KTO + keyboard wrapper producing ActionTargets |
| `model.py` | DiffusionMLP, CosineSchedule, DDIMSampler |
| `thrust_spline.py` | Clamped cubic B-spline (15 CPs, fitting + evaluation) |
| `solver.py` | Drake KTO trajectory optimization |
| `rollout_db.py` | SQLite storage for training rollouts |
| `train.py` | Training loop with CFG dropout |
| `eval.py` | Margin sweep evaluation |
| `model.pt` | Trained weights (100 epochs, 2276 frames) |
