# Outcome-Conditioned DAgger with Progressive Margin Relaxation

## Core Idea

Train a diffusion model to replace a trajectory optimizer (KTO) by iteratively
collecting on-policy data at increasing guidance margins. The diffusion model
generates position B-spline control points that are blended with the KTO
reference trajectory. As the model improves, we increase the margin (how much
we trust the model vs the optimizer), progressively transferring control from
the planner to the learned policy.

The key innovation is **outcome conditioning with relabeling**: during rollout
we always condition on `outcome=+1` (success), asking the model "what would
a successful trajectory look like?" During training, we relabel frames with
the **actual** episode outcome, teaching the model to distinguish between
actions that led to success vs failure. Combined with classifier-free guidance
(CFG), this lets the model steer toward success-conditioned trajectories at
inference time.

## Architecture

```
KTO Solver ──────────────────┐
  (trajectory optimization)  │
                             ▼
                    Position Blend: x_ref = (1-m)*kto + m*diffusion
                             │
Diffusion Model ─────────────┘
  (10 CPs x 3: x,y,theta)   │
                             ▼
                    PD Tracking Controller
                             │
                             ▼
                    Box2D Lunar Lander
```

- **KTO**: Kinematic trajectory optimization via Drake. Produces a full
  state/acceleration plan. Expensive (~3s), but deterministic per seed.
- **Diffusion Model**: MLP denoiser (DiffusionMLP) with DDIM sampling and CFG.
  Predicts noise on 30-dim output (10 control points x 3 channels: x, y, theta).
  Conditioned on 21-dim input (20 state + 1 outcome).
- **Guidance Margin** `m in [0, 1]`: Controls the blend. `m=0.001` is pure KTO
  (baseline), `m=1.0` is pure model. The PD controller always tracks velocity
  and acceleration from KTO — only position references are blended.

## Conditioning Vector (20 dims)

```
[0]      t_obs_cmd_latency (DT = 0.02s)
[1:4]    q_now (x, y, theta) — current lander pose
[4:7]    q_prev (x, y, theta) — pose at previous inference
[7:10]   obstacle (dx, dy, r) — relative obstacle (zeros for now)
[10:13]  waypoint.dq — position delta to pad
[13:16]  waypoint.dq_prime — velocity delta to target
[16:19]  guidance_q — KTO reference position at current time
[19]     action_horizon (1.5s)
[20]     outcome — CFG dim: +1 (success), -1 (fail), 0 (unconditional)
```

## The DAgger Loop

### Initialization

1. Start with a pre-trained diffusion model checkpoint
2. Initialize `margin_mean = 0.2` (conservative)
3. Collect initial episodes: 40 landed + 10 failed
4. Store in Archive DB

### Each Round

```
1. SAMPLE:    Draw N frames from Archive DB (80% from landed, 20% from failed episodes)
2. RELABEL:   Set outcome conditioning to the ACTUAL episode outcome
              (not +1 as used during rollout). This is the key DAgger insight —
              the model sees what conditioning value would have been correct.
3. TRAIN:     Fine-tune for E epochs with:
              - L2 loss on noise prediction (standard diffusion training)
              - 50% CFG dropout on outcome dim (enables classifier-free guidance)
              - Cosine noise schedule, T=100 steps
4. COLLECT:   Roll out at candidate_margin = margin_mean + 0.05
              - Always condition on outcome=+1 at inference (optimistic)
              - Apply margin with Gaussian noise: clip(|N(margin, 0.05)|, 0.001, 1.0)
              - Collect until 40 landed + 10 failed
5. EVALUATE:  Run on fixed holdout seeds (90000-90049) at multiple margins
6. ARCHIVE:   Add new episodes + frames to Archive DB
7. ADVANCE:   If collection landing rate >= 50% AND candidate margin lands >= baseline:
              margin_mean = candidate_margin
   ELSE:      Keep margin_mean, try again next round
```

### Advancement Criterion

The margin advances only when:
- The model can still land >50% of episodes at the higher margin (it's not crashing)
- The holdout eval at the candidate margin beats the baseline (m=0.001) landing count

This prevents the model from advancing when it's "getting lucky" on collection
seeds but degrading on the holdout set.

## Frame Storage Schema

Two identically-shaped SQLite databases: Archive DB (cumulative) and Current DB (per-round).

### Episodes Table
| Column | Type | Description |
|--------|------|-------------|
| uid | TEXT PK | Unique episode ID |
| git_commit | TEXT | Code version |
| env_seed | INTEGER | Environment seed (deterministic reset) |
| diffusion_seed | INTEGER | Model sampling seed (nullable) |
| walltime_start | REAL | Unix timestamp |
| outcome | INTEGER | +1 landed, -1 failed |
| sim_duration | REAL | Episode length in sim seconds |
| guidance_margin | REAL | Margin mean used for this episode |

### Frames Table
| Column | Type | Description |
|--------|------|-------------|
| uid | TEXT PK | Unique frame ID |
| episode_uid | TEXT FK | Parent episode |
| t_sim_frame | REAL | Simulation time of this frame |
| model_input | BLOB | 20-dim conditioning vector (float32) |
| model_output_cps | BLOB | 30-dim model prediction (float32) |
| actual_tracked_cps | BLOB | 30-dim KTO ground truth CPs (float32) |

Frames are collected at 3 Hz (every ~17 sim steps). A typical episode produces
~12-18 frames.

## Outcome Relabeling (Step 2)

This is the critical DAgger mechanism. During rollout:
- We condition on `outcome=+1` (success) — asking the model to produce
  successful trajectories
- The episode may actually fail (the model isn't perfect yet)

During training:
- We relabel each frame's outcome conditioning to the actual episode result
- Failed episode frames get `outcome=-1`, landed frames get `outcome=+1`
- With CFG guidance at inference, the model learns to steer *toward* +1
  trajectories and *away from* -1 trajectories

This creates a self-improving loop: failures teach the model what not to do,
successes teach what works, and the outcome conditioning gives the model a
lever to distinguish between them.

## KTO Plan Caching

The KTO solver is the bottleneck (~2-5s per episode). Since the KTO plan is
deterministic per seed (fixed initial conditions, zeroed velocities), we can:

1. Pre-solve KTO plans for all seeds in a batch (parallelized on Modal)
2. Cache the plan dict (x, y, theta, velocities, accelerations, forces)
3. Inject cached plans into the controller, skipping `warm_start()`

This provides ~6x speedup on the rollout phase. Plans include:
- Position/velocity/acceleration arrays at DT=0.02s
- Force commands (Fm, Fs)
- Number of steps and plan times

## Hyperparameter Findings

From factorial experiments (5 runs):

| What | Finding |
|------|---------|
| Data volume | Most impactful. 2000 frames/round >> 500 frames/round |
| Training intensity | Diminishing returns. 2000 epochs on 500 frames overfits |
| Learning rate | 1e-4 works for 256-dim. Lower (5e-5) doesn't help |
| Batch size | 64-128 both work. 128 needed for larger models |
| Model size | 256-dim (436K params) works well with pre-trained init. 1024/2048-dim need much more data from random init |
| Pre-trained weights | Huge advantage. Pre-trained 256 starts at 80% landing. Random init 1024/2048 start at 0-2% |

## Best Result

Run B (4x data, 2000 frames/round, 256-dim model):
- Margin reached: 0.400 (4 advances in 20 rounds)
- Holdout eval: m=0.001: 78%, m=0.4: 80%, m=0.5: 66%, m=0.7: 58%, m=1.0: 42%
- Checkpoint: `runs/B_4xdata/dagger_round20.pt`
