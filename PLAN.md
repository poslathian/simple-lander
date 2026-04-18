# Clean Restart Plan

## Rationale

The current `dp_rollout_db` has accumulated 8000+ episodes of mixed quality — old mode-collapsed plans, approximate MC returns, stale advantage labels. The code has accumulated many patches. A clean restart with all fixes baked in will produce better results.

## Pre-Restart Cleanup

Strip `t_remaining` from DP conditioning entirely (not just zero it):

1. Change `COND_DIM` from 18 to 17 (remove advantage too — use weighting only)
   - Actually keep advantage input (17 dim: obs_t(8) + obs_prev(8) + advantage(1))
   - Remove t_remaining from DPCondition entirely
2. Remove `raw_state` / `t_remaining=0` workarounds
3. Clean up dead diffusion code (_SinusoidalEmbedding, NoiseSchedule, sample_ddim, sample_ddpm)

## Phase 1: Collect Fresh KTO Data

```bash
# Create fresh DB
rm -rf dp_rollout_db_v2
.venv/bin/python3 diffusion_policy.py phase2 --db dp_rollout_db_v2 --collect 2000 \
    --critic-ckpt checkpoints/best.pt --epochs 500
```

- Collect 2000 fresh KTO episodes (up from 1000)
- Store proper MC returns from the start
- No t_remaining in states
- Train direct MLP predictor with advantage-weighted regression
- Target: match KTO stats (91% landing, return=-4.32, diverse T)

## Phase 2: Retrain Critic on Fresh Data

```bash
.venv/bin/python3 lander_critic.py train --db dp_rollout_db_v2 --epochs 300
```

- Train on clean data only
- Validate: target MAE < 1.0, corr > 0.8 on holdout
- The critic also needs t_remaining removed or standardized

## Phase 3: Verify Advantage Conditioning

```bash
.venv/bin/python3 diffusion_policy.py adv-test --seeds 100
```

- With clean advantage labels (no bias), advantage conditioning should work better
- Target: >10pp landing delta between adv=+1 and adv=-1

## Phase 4: Policy Iteration

```bash
.venv/bin/python3 diffusion_policy.py phase3 --db dp_rollout_db_v2 \
    --dp-ckpt dp_checkpoints/best.pt --critic-ckpt checkpoints/best.pt \
    --iterations 10 --episodes-per-iter 200
```

Key parameters:
- Full retrain each iteration (200-500 epochs)
- Proportional MC-return weights (not binary)
- Batch-normalized advantages for conditioning
- EMA target network (tau=0.005) for stable labels
- Checkpoint selection: use `mean_return` not just `landing_rate`
- Store proper MC returns for all new episodes

### Success Criteria

- Landing rate > 95%
- Mean return > -3.5 (better than KTO -4.32)
- Critic MAE < 1.0 throughout iterations

## Code Changes Needed

1. **Remove t_remaining from DPCondition**: Change COND_DIM to 17, update DPCondition class, remove raw_state workarounds
2. **Fix checkpoint selection**: Use `mean_return` as tiebreaker when landing rates are equal
3. **Clean up dead code**: Remove _SinusoidalEmbedding, NoiseSchedule, TIMESTEP_EMB_DIM, sample_ddim, sample_ddpm, forward_diffusion
4. **Standardize critic t_remaining**: Either remove from critic too, or always use a fixed value
5. **Rename DiffusionPolicy**: It's a direct MLP now, not a diffusion model

## Key Lessons

- The obs→plan mapping is nearly deterministic — don't use generative models
- t_remaining was a hidden data leak / train-test mismatch
- Critic bias doesn't matter for ranking but ruins absolute advantage labels
- Full retrain > conservative fine-tuning for policy iteration
- Proportional MC weights > binary advantage for weighting
- Checkpoint selection must track the metric you care about (return, not just landing rate)
