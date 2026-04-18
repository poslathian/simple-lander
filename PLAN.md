# Diffusion Policy — Continuation Plan (v2)

## Milestone 1: Clone KTO Return Stats — COMPLETE

### What Changed

Replaced the DDPM diffusion policy with a **direct MLP plan predictor** (obs → plan).
The diffusion approach mode-collapsed regardless of model size or sampling method because:

1. **`t_remaining` mismatch (root cause of mode collapse)**: The DB stored `t_remaining = max(3.0, plan_T + 1.0)` at step_idx=0, which is the plan duration, NOT a fixed timeout. At inference, code hardcoded `t_remaining=10.0`. The model learned to copy t_remaining→plan_T (corr≈1.0), so it always predicted ~9s plans at inference. **Fix**: zero `t_remaining` in DP conditioning (training and inference).

2. **Phase3 stub poisoning (critic)**: 1000 phase3 initial-state-only entries had `mc_return=0.0`, training the critic to predict V≈0 at high t_remaining. **Fix**: filter `mc_return != 0` in `RolloutDB.sample_batch`.

3. **Phase3 stub poisoning (DP)**: Same 1000 entries had mode-collapsed plans (all T≈8.9s), biasing the plan distribution. **Fix**: filter to `mc_return != 0` in `DPRolloutDB.sample_initial_states`.

4. **Diffusion is wrong for this problem**: The obs→plan mapping is nearly deterministic (a simple 22k-param MLP achieves corr=0.999, MAE=0.05 on plan duration). Diffusion's multi-step denoising added no value and caused mode collapse at high-noise timesteps.

### Current Architecture

- **Model**: Direct MLP predictor, 121k params (192-wide, 3 residual blocks, SiLU)
- **Input**: 18-dim conditioning (obs_t(8) + obs_prev(8) + t_remaining(1, always 0) + advantage(1))
- **Output**: 25-dim normalized plan encoding (12 B-spline control points × 2 + duration T)
- **Training**: MSE loss, lr=1e-3, batch_size=256, steps_per_epoch=4, 500 epochs
- **Inference**: Direct forward pass (no sampling/denoising), optional noise_scale for exploration

### Current Stats (epoch 500, 100 holdout seeds)

| Metric | DP | KTO |
|--------|----|----|
| Landing rate | 88% | 91% |
| Overall return | -5.14 | -4.32 |
| Landed return | -4.37 | -4.32 |
| T mean | 3.94 | 4.32 |
| T std | 2.20 | 1.94 |
| Early failures (<2s) | 16 | ~9 |

### Key Files

- `diffusion_policy.py` — all DP code (model, training, eval, CLI)
- `lander_critic.py` — V/Q critic with MC training, RolloutDB
- `kto_lander.py` — KTO planner + tracker (read-only)
- `checkpoints/best.pt` — critic (retrained with MC=0 filter)
- `dp_checkpoints/best.pt` — DP policy (epoch 500)
- `dp_checkpoints/norm.npz` — plan normalization stats
- `dp_rollout_db/` — 2000 episodes (1000 KTO + 1000 phase3 stubs, only KTO used for DP training)

---

## Milestone 2: Demonstrate Advantage Conditioning — IN PROGRESS

### Goal

Show that conditioning on advantage=+1 produces measurably better plans than advantage=-1. Target: >5pp landing rate delta OR >0.3 return difference.

### Plan

1. Set `advantage_dropout=0.1` (was 1.0 for unconditional BC)
2. Retrain DP with advantage labels from critic: `advantage = sign(mc_return - V(s))`
   - The retrained critic (MAE=0.73 on live rollouts) provides meaningful V(s) predictions
   - Training data has 69% positive / 31% negative advantage split
3. Run advantage A/B test: same seeds with advantage=+1 vs -1
4. Measure delta in landing rate, return, plan duration

### Commands

```bash
rm -rf dp_checkpoints
PYTHONUNBUFFERED=1 .venv/bin/python3 diffusion_policy.py phase2 \
    --db dp_rollout_db --critic-ckpt checkpoints/best.pt --epochs 500

# Then test advantage conditioning
.venv/bin/python3 diffusion_policy.py adv-test --seeds 50
```

### Risks

- With only 1 scalar advantage input, the model may not learn a strong conditional
- 1000 training samples may not be enough to learn the advantage-conditioned distribution
- If advantage doesn't work, consider: larger advantage embedding, more training data, or advantage-weighted regression instead of conditioning

---

## Milestone 3: Stable Policy Iteration

### Goal

Use a slow-moving critic to label new DP rollouts with binarized advantage, retrain the DP, and iterate. Push return stats significantly above KTO baseline.

### Plan

1. Run DP to collect new episodes (with `noise_scale > 0` for exploration)
2. Roll out each plan, compute MC return
3. Retrain critic on combined old + new data (EMA target network, tau=0.005)
4. Compute advantage = sign(MC_return - V(s)) for each episode
5. Retrain DP on expanded dataset, conditioning on advantage=+1 for above-average plans
6. Repeat 5-10 iterations, evaluating after each

### Key Parameters

- `episodes_per_iteration`: 200
- `critic_tau`: 0.005 (EMA rate for target network)
- `noise_scale`: TBD (exploration noise during DP rollouts)
- `advantage_dropout`: 0.1
- Policy update: 3 epochs at lr/10 (conservative fine-tuning)

### Commands

```bash
.venv/bin/python3 diffusion_policy.py phase3 \
    --db dp_rollout_db --dp-ckpt dp_checkpoints/best.pt \
    --critic-ckpt checkpoints/best.pt --iterations 10 --episodes-per-iter 200
```

### Success Criteria

- Landing rate > 95% (vs KTO 91%)
- Mean return > -3.5 (vs KTO -4.32)
- Critic stays calibrated (MAE < 1.5 throughout iterations)

---

## Known Issues

1. 16 early failures (<2s) on holdout — likely bad plans from model generalization gaps
2. Val loss is a poor proxy for landing rate — checkpoint selection should use holdout eval
3. `_SinusoidalEmbedding` and `NoiseSchedule` classes are still in the code but unused by the direct predictor
4. The critic uses `t_remaining=10.0` while DP uses `t_remaining=0.0` — separate CriticState instances required
5. `advantage_dropout` in DPConfig is overridden in the phase2 CLI handler — need to keep both in sync
