# Experiment Report: PPO for Lunar Lander

## Summary

Two experiments comparing RL approaches for LunarLander-v3:

1. **B-spline PPO** (plan-level, open-loop): BC pretrain + PPO fine-tune in the 25-dim B-spline plan space. Matches DirectPolicy at 95.2% landing / -4.47 return. Key finding: B-spline plans are extremely fragile — required 100x smaller exploration noise than standard RL.

2. **Native PPO** (step-level, closed-loop): Standard per-timestep PPO in the 2-dim thrust action space. Achieves 89.8% landing / -3.28 return. Faster trajectories when landing succeeds, but less reliable than open-loop planners. Sparse reward causes PPO instability.

### 500-Seed Results (seeds 5000-5499)

| Metric | KTO (solver) | DirectPolicy | B-spline PPO | Native PPO |
|---|---|---|---|---|
| Landing rate | 94.0% | 95.4% | **95.2%** | 89.8% |
| Mean return | -4.76 | -4.48 | -4.47 | **-3.28** |
| Planning time | ~100ms | ~0.3ms | ~0.3ms | N/A (online) |
| Closed-loop | No | No | No | **Yes** |

---

## Experiment 1: B-spline PPO

### Problem Setup

Single-step (bandit-like) RL: the policy observes the initial state after warmup, outputs a 25-dim B-spline plan encoding (12 control points x 2 + duration), which is executed open-loop via a PD tracker. The episode return is the sole reward signal:
- Landed: `-t_elapsed` (faster is better, typically -2 to -6)
- Failed: `-(t_elapsed + 10)` (heavy penalty)

### Architecture

- **Policy**: MLP (16 -> 192 -> 192 -> 25), 83K params. LayerNorm + SiLU, 3 layers.
- **Value**: MLP (16 -> 192 -> 192 -> 1), 79K params. Same architecture.
- **Input**: obs_t(8) + obs_prev(8) = 16-dim.
- **Total**: 162K params (budget: 623K).

### Phase 1: Behavioral Cloning (BC)

Random plans in 25D never produce valid trajectories. Pure RL from scratch produces zero gradient signal (all episodes crash identically at return = -10.02).

- Collected 2000 KTO episodes (93% landing rate)
- Trained 300 epochs with cosine LR schedule (1e-3 -> 0)
- Final BC loss: 0.000607 (normalized MSE)
- BC eval: **87% landing / -5.22 return** (100 seeds)

### Phase 2: PPO Fine-Tuning

#### The Fragility Problem

Initial PPO with standard exploration noise (log_std = -0.5 to -1.5, std ~ 0.2-0.6) produced 0% landing across all iterations. Even small Gaussian noise on B-spline control points destroys trajectories because:
1. Adjacent control points must maintain smooth spatial relationships
2. Resulting accelerations must stay within the PD tracker's feasible envelope
3. Flatness-derived attitude must remain within e-stop bounds

#### Solution: Tiny Exploration Noise

Setting log_std = -4.0 (std ~ 0.018 in normalized space) solved the problem:
- ~60-80% of sampled plans still land (enough for gradient signal)
- ~20-40% fail (enough variance for advantage estimation)

#### Training

- 500 total PPO iterations (200 + 300 continuation)
- 200-300 episodes per iteration, LR 3e-5 / 1e-5
- Result: **95.2% landing / -4.47 return** (500 seeds)

---

## Experiment 2: Native PPO (Step-by-Step)

### Problem Setup

Standard closed-loop RL: at each timestep (50 Hz), the policy observes the 8-dim gymnasium observation and outputs 2-dim continuous actions [main_engine, side_thruster] in [-1, 1]. Same environment setup (warmup + randomize), same sparse reward function.

### Architecture

- **Policy**: MLP (8 -> 256 -> 256 -> 2), 136K params. LayerNorm + Tanh, 3 layers. Tanh-squashed Gaussian.
- **Value**: MLP (8 -> 256 -> 256 -> 1), 136K params. Same architecture.
- **Total**: 272K params.

### Training

- 5M timesteps, 4096 steps/rollout, 10 PPO epochs
- GAE with gamma=1.0, lambda=0.95
- LR 3e-4, clip epsilon 0.2

### Results

Training progressed rapidly from 0% to ~94% landing (100-seed eval) by ~1.5M steps. But then the policy became unstable:

| Steps | Eval Landing | Eval Return | Notes |
|---|---|---|---|
| 200K | 50% | -6.50 | Early learning |
| 800K | 92% | -3.92 | Rapid improvement |
| 1.5M | 94% | -2.93 | Peak performance |
| 2.0M | 94% | -3.13 | Best saved checkpoint |
| 2.5M | 84% | -4.60 | Degradation begins |

Stabilization attempt (LR 5e-5, clip 0.1 from best checkpoint) did not prevent oscillation — eval continued bouncing 81-93%.

Final 500-seed eval of best checkpoint: **89.8% landing / -3.28 return**.

### Why Native PPO Is Unstable

The sparse reward (0 at every step except terminal) combined with gamma=1.0 means:
- V(s) must predict the distant terminal outcome from every mid-flight state
- GAE advantages are noisy because V(s) errors propagate across 100-300 step episodes
- The policy oscillates: small policy changes cause large trajectory divergence, leading to different outcomes, which cause large value function updates, which cause large policy updates

The B-spline approach avoids this by reducing the problem to a single decision point with immediate feedback.

---

## Conclusions

### 1. Action space determines the reliability/speed tradeoff

- **B-spline (open-loop)**: Higher reliability (95.2%) but slower trajectories (-4.47). The PD tracker provides robustness guarantees once a feasible plan exists.
- **Native (closed-loop)**: Faster trajectories (-3.28) but lower reliability (89.8%). No tracker to fall back on — every timestep decision must be correct.

### 2. B-spline plans are extremely fragile under perturbation

Standard RL exploration (std > 0.1) produces 0% valid plans. Required std ~ 0.018 (100x smaller). This makes RL in plan space fundamentally constrained — the "useful" region of action space is a razor-thin manifold in 25D.

### 3. Sparse reward is the bottleneck for step-level RL

With dense reward shaping (distance to pad, velocity penalties, etc.), native PPO would likely achieve both higher landing rate AND faster trajectories. The sparse reward (-t on land, -t-10 on crash) is well-suited for plan-level single-step RL but poorly suited for 300-step credit assignment.

### 4. BC pretraining is mandatory for plan-level RL, unnecessary for step-level

Plan-level RL cannot bootstrap from random — BC is required. Step-level RL in the native action space learns from scratch (23% landing by 4K steps), because 2D thrust actions have immediate physical effects that exploration can discover.

## Files

- `ppo_lander.py` — B-spline PPO: BC pretraining + RL fine-tuning
- `ppo_native.py` — Native step-by-step PPO
- `kto_lander.py` — Environment infrastructure (from simple-lander-reboot)
- `checkpoints/best.pt` — Best B-spline PPO checkpoint (95.2% landing)
- `checkpoints_native/best.pt` — Best native PPO checkpoint (89.8% landing)
