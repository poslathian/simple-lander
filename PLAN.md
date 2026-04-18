# Diffusion Policy — Continuation Plan

## What Exists

Three committed files implementing the full pipeline:

- **`kto_lander.py`** — Drake KTO trajectory optimizer + PD tracker (working, ~91% landing rate)
- **`lander_critic.py`** — V/Q critic with MC training, RolloutDB (working)
- **`diffusion_policy.py`** — DDPM/DDIM diffusion policy, DPRolloutDB, DPTrainer, CLI

The DP is a 292k-param noise-prediction MLP (256-wide, 4 residual blocks, SiLU, sinusoidal timestep embedding). 10-step DDIM inference. Outputs 25-dim plan (12 B-spline control points × 2 + duration T). Plan reconstruction via Drake BsplineTrajectory is verified exact (<1e-9 roundtrip error).

## Current State

**Phase 2 (behavioral cloning) works well:**
- 1000 KTO episodes in `dp_rollout_db/` (201k transitions, 500k capacity)
- DP trained on initial-state transitions only (step_idx=0)
- **91% landing rate** on holdout seeds — matches KTO
- Loss converged: ~0.025 train, ~0.018 val

**Phase 3 (online improvement) has issues:**
- Policy updates consistently degrade holdout performance (92% → ~60-86%)
- Reduced to 3 epochs at LR/10 — better than 20 epochs but still no improvement over baseline
- Checkpoint revert mechanism works (restores best on degradation)

**Advantage conditioning does NOT work yet:**
- `adv-test` shows no meaningful difference between advantage=+1 and advantage=-1
- Root cause identified: **the critic checkpoint (checkpoints/best.pt) is stale** — it was trained on only the original 200-episode `rollout_db/`, not the 1000-episode `dp_rollout_db/`
- Critic eval shows: correlation=-0.008, MAE=8.03, bias=+8.03 (predicts -0.54 when actual returns are -8.57)
- Since advantage = sign(MC_return - V(s)), and V(s) ≈ -0.54 while all returns are << -0.54, advantage labels are ALL -1 → no signal

## What To Do Next

### Step 1: Retrain the critic on the 1000-episode DB

```bash
.venv/bin/python3 lander_critic.py train --db dp_rollout_db --epochs 200
```

Then validate:
```bash
.venv/bin/python3 diffusion_policy.py critic-eval --seeds 50
```

Target: correlation > 0.7, MAE < 2.0. The critic must accurately rank states before advantage conditioning can work.

### Step 2: Retrain DP with good advantage labels

```bash
rm -rf dp_checkpoints
.venv/bin/python3 diffusion_policy.py phase2 --db dp_rollout_db --critic-ckpt checkpoints/best.pt --epochs 200
```

Then test advantage conditioning:
```bash
.venv/bin/python3 diffusion_policy.py adv-test --seeds 50
```

Target: measurable delta between adv=+1 and adv=-1 (>5pp landing or >0.3 return difference).

### Step 3: Phase 3 online improvement

```bash
.venv/bin/python3 diffusion_policy.py phase3 --db dp_rollout_db --dp-ckpt dp_checkpoints/best.pt --critic-ckpt checkpoints/best.pt --iterations 10 --episodes-per-iter 200
```

Phase 3 now includes:
- **Target network EMA** (tau=0.005) — soft-updates critic target after each iteration
- **Critic eval** — validates V(s) vs actual MC returns each iteration
- **3 epochs at LR/10** — conservative policy updates to prevent catastrophic forgetting
- **Checkpoint revert** — restores best checkpoint on holdout degradation
- **Initial-state-only DP storage** — only stores step_idx=0 for DP episodes (saves ~200x DB capacity)

### Step 4: Final comparison

```bash
.venv/bin/python3 diffusion_policy.py compare --episodes 100
```

## Architecture Notes

### Advantage conditioning flow
- Training: `advantage = sign(MC_return - V(s))` — did this plan beat the critic's prediction?
- 10% advantage dropout (mask to 0) during training — classifier-free guidance
- Inference: always condition on advantage=+1 ("generate a good plan")

### Phase 3 structured replay sampling (500 transitions/batch)
| Bucket | Count | Selection |
|--------|-------|-----------|
| Newest | 100 | Most recent episode IDs |
| Worst recent | 100 | Lowest advantage in last 10 iterations |
| Edge cases | 100 | Highest tracking error + failures |
| Best overall | 100 | Highest advantage across all time |
| Best recent | 100 | Highest advantage in last 10 iterations |

### HER
Relabels advantage conditioning to match actual achieved advantage with probability 0.5.

### CLI subcommands
- `phase2` — behavioral cloning from KTO
- `phase3` — online improvement loop
- `evaluate` — holdout evaluation (100 seeds)
- `adv-test` — advantage conditioning A/B test
- `critic-eval` — critic V(s) vs actual MC returns
- `run [--render]` — single DP episode
- `compare` — side-by-side KTO vs DP

## Key Files
- `diffusion_policy.py` — all DP code (~1500 lines)
- `diffusion_policy.pyi` — type stub
- `lander_critic.py` — critic + rollout collection
- `kto_lander.py` — KTO planner + tracker (read-only, not modified)
- `dp_rollout_db/` — 1000 KTO episodes, 500k capacity
- `rollout_db/` — original 200-episode DB (stale, don't use)
- `checkpoints/best.pt` — **STALE critic** trained on 200 episodes, needs retraining
- `dp_checkpoints/best.pt` — DP policy checkpoint
- `dp_checkpoints/norm.npz` — plan normalization stats

## Known Issues
1. `checkpoints/best.pt` critic is stale — trained on 200 episodes, not 1000. **Must retrain before anything else.**
2. `lander_critic.py train` CLI uses `--db rollout_db` by default — pass `--db dp_rollout_db` explicitly
3. Phase 3 critic trainer uses `CriticTrainer` which trains on ALL transitions (not just step_idx=0) — this is correct for the critic but means critic training is slower with the full DB
4. The `_load_critic` helper in diffusion_policy.py uses a pickle shim (`_make_pickle_helper`) to handle `CriticConfig`/`TrunkType` deserialization when running as `__main__`
