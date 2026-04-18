# Experiment Report: Diffusion Policy for Lunar Lander

## Summary

Attempted to train a learned policy (originally diffusion-based, later direct MLP) to clone and improve upon KTO trajectory optimization for the LunarLander-v3 environment. Achieved 93% landing rate (vs KTO 91%) but returns remain worse (-6.45 vs KTO -4.32).

## Timeline of Key Findings

### Bug Fixes (Critical)

1. **MC=0 poisoning in critic training**: 1000 phase3 initial-state-only entries had `mc_return=0.0`, training the critic to predict V~0. Fix: filter `mc_return != 0` in `sample_batch`.

2. **`t_remaining` mismatch (root cause of mode collapse)**: The DB stored `t_remaining = max(3.0, plan_T + 1.0)` at step_idx=0 — this is the plan duration, NOT a timeout. At inference, code used `t_remaining=10.0`. The model learned `t_remaining → plan_T` (corr=1.0), so always predicted ~9s plans. Fix: zero `t_remaining` in DP conditioning.

3. **Phase3 stub plan poisoning**: 1000 phase3 entries had mode-collapsed plans (all T~8.9s) mixed with 1000 diverse KTO plans. Fix: filter to `mc_return != 0` in `sample_initial_states`.

4. **Inverted advantage labels**: Critic V(s) computed on zeroed-t_remaining states (wrong) while trained on real t_remaining. V=-3.24 vs MC=-4.91 → 75% negative labels (inverted). Fix: pass `raw_state` to critic, zeroed state to DP.

5. **Phase3 episodes didn't store MC returns**: Initial-state-only storage skipped terminal reward. Fix: set terminal reward on initial transition before storing.

### Architecture Decisions

1. **Diffusion → Direct MLP**: Diffusion noise prediction mode-collapsed regardless of model size (13k-292k params), step count (10-100), or sampling method (DDIM, DDPM, stochastic). Root cause: the obs→plan mapping is nearly deterministic (simple MLP achieves corr=0.999). Diffusion's multi-step denoising added no value. Direct MLP prediction works perfectly.

2. **Timestep embedding removed**: 32-dim sinusoidal embedding was half the input space and unnecessary. Replaced with 1-dim scalar (then removed entirely with direct predictor).

3. **Model size**: 121k params (192-wide, 3 residual blocks). Input: 18-dim (obs_t + obs_prev + t_remaining + advantage). Output: 25-dim normalized plan.

### Advantage Conditioning

- **Scalar conditioning failed**: Adding advantage as 1 dim to 18-dim input — model ignored it (1 dim vs 16 obs dims).
- **Advantage-weighted regression works**: Weight MSE loss by advantage. Good plans get weight 2, bad plans 0.1. Produces measurable delta: adv=+1 87% landing vs adv=-1 74% (+13pp).
- **Proportional MC-return weights**: Better than binary. Weight 0.1-2.0 scaled by return within batch.
- **Batch-normalized advantages**: `sign(mc - batch_mean)` instead of `sign(mc - V(s))`. Removes critic bias.

### Policy Iteration

- **Conservative fine-tuning fails**: 3 epochs at lr/10 never improved beyond 92%. Model always reverted.
- **Full retrain works**: 200 epochs from scratch each iteration. Consistently hits 92-93%.
- **Landing rate improves, returns don't**: 93% > KTO 91%, but return -6.45 << KTO -4.32. The DP produces plans that land but take longer or have worse trajectories.
- **Critic bias**: Systematic -3.2 bias makes advantage labels noisy. Correlation decent (0.70) but absolute values wrong.

## Current State

| Metric | DP (best) | KTO baseline |
|--------|-----------|-------------|
| Landing rate | **93%** | 91% |
| Overall return | -6.45 | **-4.32** |
| Landed return | ~-4.5 | **-4.32** |
| Plan T mean | 3.9s | 4.3s |
| Plan T std | 2.2s | 1.9s |
| Model params | 121k | N/A (optimizer) |

## Key Problems Remaining

1. **Return gap**: DP returns (-6.45) much worse than KTO (-4.32) despite higher landing rate. The 16 early failures (<2s) contribute -10 each, dragging the mean.
2. **Critic accuracy**: MAE=3.93 on holdout. Bias=-3.92 (systematic pessimism). Advantage labels are noisy.
3. **DB contamination**: The dp_rollout_db has accumulated stale data from many experiments — old mode-collapsed plans, approximate MC returns, mixed data quality.
4. **`t_remaining` is vestigial**: Always zeroed for DP, always 10.0 for critic. Should be removed from DP conditioning entirely.

## Files Modified

- `diffusion_policy.py` — major rewrite: diffusion→direct MLP, advantage weighting, t_remaining fixes, phase3 full retrain
- `lander_critic.py` — MC=0 filter in `sample_batch`
