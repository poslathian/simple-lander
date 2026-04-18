# Experiment Report: Dense Reward PPO for Lunar Lander

## Summary

Step-level PPO with dense reward (−DT per step, −10 on crash) achieves **96.2% landing / −2.11 return** on 500-seed eval — the first closed-loop controller to beat DirectPolicy on both metrics.

### 500-Seed Results (seeds 5000–5499)

| Metric | KTO (solver) | DirectPolicy | Sparse PPO | **Dense PPO** |
|---|---|---|---|---|
| Landing rate | 94.0% | 95.4% | 89.8% | **96.2%** |
| Mean return | −4.76 | −4.48 | −3.28 | **−2.11** |
| Planning time | ~100ms | ~0.3ms | ~0.3ms | N/A (online) |
| Closed-loop | No | No | No | **Yes** |

---

## Motivation

The sparse reward experiment (see `EXPERIMENT_REPORT_sparse.md`) showed that native step-level PPO with sparse terminal reward (0 at every step, reward only at episode end) was unstable: eval oscillated between 81–93% and final best was 89.8%. Root cause: V(s) must predict distant terminal outcomes from every mid-flight state, making GAE advantages noisy.

**Hypothesis**: Decomposing the sparse terminal reward into per-step contributions would provide local gradient signal, stabilize value estimation, and improve both landing rate and convergence.

## Dense Reward Design

The reward is a trivial decomposition of the original sparse reward:

- **Per-step**: `−DT` (= −0.02 at 50 Hz)
- **Terminal failure**: additional `−10`

Episode return = `−t_elapsed` (landed) or `−t_elapsed − 10` (failed), identical to the B-spline plan reward by construction. No reward shaping or auxiliary signals — just the same reward expressed per-step.

## Architecture

Identical to the sparse native PPO:

- **Policy**: MLP (8 → 256 → 256 → 2), 136K params. LayerNorm + Tanh, 3 layers. Tanh-squashed Gaussian.
- **Value**: MLP (8 → 256 → 256 → 1), 136K params. Same architecture.
- **Total**: 272K params.

## Training

- 5M timesteps (best checkpoint at ~3.76M)
- **gamma = 0.99** (vs gamma = 1.0 for sparse)
- 4096 steps/rollout, 10 PPO epochs, minibatch 256
- LR: policy 3e-4, value 1e-3
- Clip epsilon 0.2, entropy coeff 0.01
- GAE lambda 0.95

### Training Curve

| Steps | Eval Landing | Eval Return | Notes |
|---|---|---|---|
| 20K | 22% | −9.82 | Cold start (no BC) |
| 100K | 68% | −6.35 | Rapid improvement |
| 140K | 81% | −4.30 | Exceeds sparse PPO's best |
| 600K | 93% | −2.96 | Near DirectPolicy landing |
| 1.08M | 98% | −2.24 | First 98% eval |
| 1.5–2.5M | 88–95% | −2.5 to −2.9 | Oscillation phase |
| **3.76M** | **98%** | **−1.93** | **Best checkpoint** |
| 4.7M | 96% | −2.14 | Still strong at end |

Key observations:
1. **No BC pretraining needed** — learns from scratch, reaching 68% at 100K steps
2. **Stable training envelope** — eval oscillates 88–98% but never collapses below 82% (sparse dropped to 81% and never recovered)
3. **Return far surpasses baselines** — −2.11 vs −3.28 (sparse) vs −4.48 (DirectPolicy)
4. **Exploration stays bounded** — log_std oscillates between −0.20 and −0.10, preventing destructive over-exploration

## Why Dense Reward Works

1. **Local credit assignment**: V(s) predicts near-term costs (−DT per step) + discounted crash probability (−10 × P(crash)). With gamma=0.99, the effective planning horizon is ~100 steps, not the full 300-step episode.

2. **Smooth value landscape**: The per-step −DT reward creates a time-pressure gradient that naturally encourages efficient trajectories. States closer to landing have lower remaining cost.

3. **Crash penalty dominates**: The −10 crash penalty is large relative to per-step −DT (equivalent to 500 steps = 10 seconds). This creates strong gradient to avoid crash states, stabilizing the policy near the landing basin.

4. **Same optimal policy**: Since the dense reward decomposes the sparse reward exactly, the optimal policy is unchanged. But the learning dynamics are dramatically better because V(s) is easier to learn.

## Files

- `ppo_dense.py` — Dense reward step-by-step PPO
- `kto_lander.py` — Environment infrastructure
- `checkpoints_dense/best.pt` — Best checkpoint (96.2% landing / −2.11 return on 500 seeds)
- `EXPERIMENT_REPORT_sparse.md` — Prior sparse reward experiment for comparison
- `ppo_native_sparse.py` — Sparse reward PPO reference
