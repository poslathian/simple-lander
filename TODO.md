# TODO

## Refactor: supervision CPs from observed trajectory, not internal blend

The guidance margin mechanism (weighted average of KTO + diffusion) is currently duplicated in:
- `diffusion_controller.py` get_action()
- `dagger_loop.py` collect_episode()
- `modal_rollout.py` rollout_with_cache() (remote)
- `modal_rollout.py` rollout_with_cache_local()

Each copy manually reconstructs what the controller tracked. This is fragile and led to bugs (clamp vs blend mismatch, stale code in modal_rollout).

**Proposed fix:** Move the entire margin mechanism inside `KTODiffusionController`. The supervision CPs for DAgger should come from fitting a new B-spline to the *observed trajectory* (actual lander positions over the action horizon), not by reaching inside the controller for whatever KTO+diffusion blend it computed.

Benefits:
1. **Single source of truth** — margin logic lives in one place
2. **Incorporates PD controller output** — we learn to generate plans the PD controller can actually follow, not idealized blends it may not track accurately
3. **No internal state leaking** — collect_episode doesn't need `_last_cps_norm`, `_kto`, `_diff_splines` etc.
4. **Simpler DAgger loop** — just run episode, fit CPs to what actually happened, label with outcome

The controller would expose something like `get_supervision_cps(obs_trajectory)` that fits CPs to the observed positions over the last action horizon window.

## Fix: t_obs_cmd_latency is wrong

Currently hardcoded to `DT` (0.02s). This should be the actual model inference time (~10-50ms), not the sim timestep or the inference period (333ms at 3Hz). The 333ms between inference calls is how long we wait, not how long the plan takes to produce. The model returns CPs almost instantly.

## Finding: Outcome conditioning as OOD detector

When the model is in-distribution, outcome=+1 consistently lands more than outcome=-1 (typical gap: +10 to +25%). When the model is out of distribution (e.g., margin too high for its current ability), the gap **inverts** — outcome=-1 lands more than +1.

This makes the outcome conditioning gap a free out-of-distribution detector:
- **Gap > 0**: model is in-distribution, outcome conditioning is working
- **Gap ≈ 0**: model is at the edge, conditioning has no signal
- **Gap < 0**: model is OOD, its "success" predictions are worse than "failure"

This could be used at inference time: if outcome=-1 would produce better trajectories than outcome=+1, the model shouldn't be trusted at the current margin. The margin should be reduced until the gap is positive again.

Observed during DAgger training on 256-hidden model:
- R11 at m=0.03: +1=11%, -1=25% (gap=-14, OOD → crashed)
- R15 at m=0.03: +1=67%, -1=50% (gap=+17, in-distribution → survived)
- R19 at m=0.03: +1=75%, -1=65% (gap=+10, healthy → advanced to m=0.04)
