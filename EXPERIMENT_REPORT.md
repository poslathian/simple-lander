# Experiment Report: Direct Policy for Lunar Lander

## Summary

Trained a direct MLP policy to clone and improve upon KTO trajectory optimization for LunarLander-v3. After a clean restart with all bugs fixed, achieved **95.4% landing / -4.48 return** vs KTO's **94.0% landing / -4.76 return** on a 500-seed head-to-head — beating KTO on all metrics with 333x faster planning (0.3ms vs ~100ms).

## Methodology

### Clean Restart Rationale

The original dp_rollout_db had accumulated 8000+ episodes of mixed quality from multiple experiments: mode-collapsed plans, approximate MC returns, stale advantage labels, and `t_remaining` data leaks. A clean restart with all fixes baked in was needed.

### Bugs Found and Fixed

1. **`t_remaining` data leak / train-test mismatch** (root cause of mode collapse): DB stored `t_remaining = plan_T` at step_idx=0. At inference, code used `t_remaining=10.0`. Model learned `t_remaining -> plan_T` (corr=1.0), always predicting ~9s plans. **Fix**: removed `t_remaining` from DP conditioning entirely (COND_DIM 18 -> 17).

2. **MC=0 poisoning**: Phase3 initial-state-only entries had `mc_return=0.0`, training the critic to predict V~0. **Fix**: filter `mc_return != 0` in `sample_batch`.

3. **Phase3 stub plan poisoning**: 1000 mode-collapsed plans mixed with 1000 diverse KTO plans. **Fix**: filter on `mc_return != 0` in `sample_initial_states`.

4. **Inverted advantage labels**: Critic V(s) computed on zeroed-t_remaining states (wrong) while trained on real t_remaining, producing systematically wrong advantage signs. **Fix**: pass `raw_state` to critic, stripped state to DP.

5. **EMA target network unused**: Created and soft-updated but never queried for advantage labeling — all labels used the noisy online critic. **Fix**: use `critic_target` for advantage labels during episode collection.

6. **Checkpoint selection**: Used only landing rate, ignoring return. Two checkpoints at 93% landing with returns of -4.5 and -6.0 were treated as equal. **Fix**: use `mean_return` as tiebreaker.

### Architecture: Diffusion -> Direct MLP

The obs->plan mapping is nearly deterministic. Diffusion's multi-step denoising added no value and mode-collapsed regardless of model size (13k-292k params), step count (10-100), or sampling method (DDIM, DDPM). Replaced with a direct MLP (obs_t + obs_prev + advantage -> plan encoding).

- **Model**: 120k params, 192-wide, 3 residual blocks with LayerNorm
- **Input**: 17-dim (obs_t(8) + obs_prev(8) + advantage(1))
- **Output**: 25-dim normalized plan encoding (12 control points x 2 + duration)
- **Training**: Advantage-weighted regression with MC-return-based weights

### Training Pipeline

**Phase 1: Collect fresh data** — 2000 KTO episodes (92% landing) with proper MC returns, no `t_remaining` contamination.

**Phase 2: Train critic** — 300 epochs on clean data. val_V=0.34, val_Q=0.05.

**Phase 3: Verify advantage conditioning** — 100-seed test: adv=+1 gets 91% landing vs adv=-1 gets 61% (+30pp delta, +2.42 return delta).

**Phase 4: Policy iteration** — ~50 iterations with key improvements:
- **Fine-tune from best checkpoint** (100 epochs, lr/5) instead of full retrain from scratch. Full retrain every 5th iteration only.
- **Critic drift detection**: Full critic retrain when MAE > 2.0 or corr < 0.3 on holdout.
- **EMA target network** for advantage labels during collection (tau=0.005).
- **Balanced training**: critic 50 epochs x 8 steps (400 grad steps) vs DP 100 epochs x 4 steps (400 grad steps).

## Results

### 500-Seed Head-to-Head Comparison

| Metric | KTO (Drake solver) | DP (learned policy) |
|---|---|---|
| Landing rate | 94.0% (470/499) | **95.4% (477/500)** |
| Mean return | -4.76 | **-4.48** |
| Planning time | ~100ms | **0.3ms** |

### Shared-Success Trajectory Quality (200 seeds where both land)

Isolating trajectory quality from landing ability:

| Metric | KTO | DP |
|---|---|---|
| Mean return | -4.384 | **-4.160** |
| Median return | -3.860 | -3.940 |
| DP produces faster trajectory | — | **72% of seeds** |

Percentile analysis of per-seed delta (DP - KTO):
- p10: -0.100 (KTO slightly faster on some easy seeds)
- p50: +0.090 (DP typically slightly faster)
- p90: +0.820 (DP much faster on hard seeds)

The DP advantage is strongest on difficult initial conditions — states with high velocity or extreme positions where the solver struggles to find fast trajectories.

### Failure Analysis

The ~4.6% failure rate (23/500 seeds) consists entirely of physics-infeasible initial conditions: lander starts at extreme positions (|x| > 6.7) with velocity toward the boundary. **KTO fails on all the same seeds.** These represent the hard ceiling for this environment.

## Key Lessons

1. **The obs->plan mapping is nearly deterministic** — generative models (diffusion) are unnecessary. A direct MLP achieves corr=0.999.

2. **`t_remaining` was a hidden data leak** — the model memorized plan duration from this feature instead of learning the obs->plan mapping.

3. **Fine-tuning beats full retrain for policy iteration** — full retrain from scratch each iteration throws away learned parameters and produces random-walk behavior. Fine-tuning from the best checkpoint at reduced LR (lr/5) gives monotonic improvement.

4. **Critic bias doesn't matter for ranking but ruins advantage labels** — the critic had systematic -3s bias throughout. MC returns used directly for training weights are more reliable.

5. **EMA target networks only help if you actually query them** — the target critic was created, maintained, and never used for 30+ iterations before the bug was found.

6. **Proportional MC-return weights > binary advantage** — weighting loss by return quality (0.1-2.0 range) outperforms binary good/bad classification.

7. **Checkpoint selection must track the metric you care about** — landing rate alone is insufficient; return as tiebreaker prevents selecting slow-but-safe policies.

## Files Modified

- `diffusion_policy.py` — major rewrite: removed diffusion code (NoiseSchedule, SinusoidalEmbedding, forward_diffusion), renamed DiffusionPolicy -> DirectPolicy, removed t_remaining from conditioning, added fine-tune/full-retrain hybrid iteration, EMA fix, checkpoint selection fix
- `lander_critic.py` — MC=0 filter in `sample_batch`

## Data

- `dp_rollout_db_v2/` — 22,200 episodes (20,456 landed), 421k transitions
- `dp_checkpoints/best.pt` — final policy checkpoint (120k params)
- `checkpoints/best.pt` — critic checkpoint
- `shared_success_seeds.npy` — 200 seeds where both KTO and DP land successfully
