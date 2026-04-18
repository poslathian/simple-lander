# Dense Reward PPO for Lunar Lander

## Prior Work: Sparse Reward PPO (simple-lander-rl)

The sparse reward experiment (`EXPERIMENT_REPORT_sparse.md`) tested two PPO variants:

1. **B-spline PPO** (open-loop, plan-level): 95.2% landing / -4.47 return. Required BC pretraining and extremely tiny exploration noise (std~0.018). Matches DirectPolicy but adds nothing over it.

2. **Native PPO** (closed-loop, step-level): 89.8% landing / -3.28 return. Faster trajectories when landing, but unstable training. The policy oscillated between 81-94% landing during training and never converged.

**Root cause of instability**: Sparse reward (0 at every step, terminal reward only) combined with gamma=1.0 means the value function must predict distant terminal outcomes from every mid-flight state. GAE advantages are noisy because V(s) errors propagate across 100-300 step episodes. Small policy changes cause large trajectory divergence, which causes large value updates, creating a feedback loop.

## Hypothesis

Dense per-step reward shaping will:
1. **Stabilize training** by providing local gradient signal at every timestep, so V(s) only needs to predict near-term returns
2. **Enable gamma < 1** since rewards aren't concentrated at a single terminal step
3. **Improve final performance** beyond 89.8% landing — the sparse PPO showed the policy *can* reach 94% but can't hold it. Dense rewards should make that stable.
4. **Match or beat DirectPolicy** (95.4% landing / -4.48 return) as the first closed-loop controller to do so.

## Dense Reward Design

Decompose the sparse terminal reward into per-step contributions:

- **Per-step reward**: `-DT` (= -0.02 at 50 Hz) every timestep
- **Terminal failure penalty**: additional `-10` on crash/timeout

**Episode return by construction**:
- Landed after N steps: `N * (-DT) = -t_elapsed` (matches B-spline sparse reward exactly)
- Failed after N steps: `N * (-DT) - 10 = -t_elapsed - 10` (matches B-spline sparse reward exactly)

This is not reward shaping — it's the *same* reward function, just expressed per-step instead of terminally. The value function now has a clear local signal: every step costs -DT, and crashing costs -10 extra. V(s) only needs to predict near-term costs + probability-weighted crash penalty, instead of predicting the entire distant terminal outcome from mid-flight.

## Execution Plan

1. Fork `ppo_native.py` to `ppo_dense.py`
2. Implement dense reward function (start with potential-based)
3. Set gamma=0.99 (standard for dense reward)
4. Train 5M steps with 100-seed eval every 20K steps
5. If unstable, try heuristic dense reward instead
6. Final 500-seed eval comparison against all baselines

## Success Criteria

- Landing rate >= 95% (match DirectPolicy)
- Stable training (no oscillation)
- Closed-loop advantage: faster trajectories than open-loop planners
