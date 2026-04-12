# PLAN.md — SDEdit-style diffusion guidance (replaces weighted-average blend)

This document is the resume point for the next Claude session. The user will
clear context and resume from here. Be self-sufficient.

---

## Where things stand right now

### Phase 4 dagger run is live in the background

- **Background task ID**: `bo7oskvxu` (the active python process). Tail with:
  ```
  tail -50 /private/tmp/claude-503/-Users-canopy4-habitat3-scratchpad-critic/e819d92d-703b-4c41-b831-793af1f757c3/tasks/bo7oskvxu.output
  ```
  Confirm alive with: `pgrep -lf "python.*dagger_advantage" | head -3`
- **Cron**: `e14b4fec` fires every 5 min on minutes :03/:08/:13/.../:58 to do
  a check-in against `bo7oskvxu.output`. Session-only. Reporting only — does
  not modify the run.
- **Run command** (currently in flight; do not restart unless necessary):
  ```
  /Users/canopy4/habitat3/.venv/bin/python -u -m dagger_advantage \
    --resume --resume-round 30 \
    --starting-checkpoint ../phase4_results/advantage_round30.pt \
    --critic-db ../phase2_results/diverse_rollouts_v2.db \
    --critic-uid 640fdf61-372 \
    --rounds 70 --initial-margin 0.1440 \
    --margin-mode geometric --margin-advance-factor 1.2 \
    --margin-retreat-factor 0.8 --margin-floor 0.01 \
    --target-landed 40 --target-failed 10 \
    --train-frames 500 --train-epochs 200 --batch-size 32 --lr 5e-5 \
    --hidden 256 --n-blocks 6 \
    --smart-sampling --sample-recent-frac 0.20 --sample-best-frac 0.40 \
      --sample-edge-frac 0.10 --sample-random-frac 0.30 \
      --recent-window-episodes 100 --best-top-k-episodes 200 \
    --alarm-v-mae 0.20 --alarm-sep 0.0 --alarm-rounds 2 \
    --retarget-mode polyak --retarget-update-interval 1 --retarget-tau 0.05 \
    --retarget-recent-episodes 200 --retarget-epochs 20 \
    --run-dir ../phase4_results --seed-offset 1600000
  ```
- **Latest progress** (as of writing): around round 43-44, m oscillating
  0.13–0.21, m=1.0 holdout still 0/10. Plateau at m≈0.18 broken at R40-41
  to m≈0.21, retreating and re-advancing in 1-step cycles. Live critic
  v_mae~0.12, target v_mae~0.14-0.16, target acc 80-81%, sep 0.02-0.05.
- **Checkpoints saved**: `advantage_round{5,10,15,20,25,30,35,40}.pt` in
  `../phase4_results/`. New ones every 5 rounds.
- **Sprite plots auto-pushed** at every checkpoint save:
  - https://critic-plot-bnmbj.sprites.app/margin_curve.png
  - https://critic-plot-bnmbj.sprites.app/critic_quality.png
  - Push script: `../push_plot.sh`. Persistent service `plot-server`
    (`python3 -m http.server 8080`) on sprite `critic-plot`.

### Critic state
- Critic uid `640fdf61-372` lives in `../phase2_results/diverse_rollouts_v2.db`
  in the `critics` table. Loaded at startup as the **target** critic.
- A **live critic** is cloned at startup (always-on Polyak τ=0.05) and trained
  every round on the most recent 200 episodes' frames using their hindsight
  V_target labels. Polyak blends back into target every round.
- Per-round critic quality is recorded into the dagger archive's
  `round_critic_quality` table by `record_round_critic_quality`.

### What's working / what's not
- **Working**: smart sampling (PR2), per-round critic quality monitoring
  (PR1), breakdown alarm (PR3, currently informational), Polyak retargeting
  (PR3, always-on per user direction).
- **Plateau**: margin oscillates 0.13–0.21. The dagger loop keeps advancing
  and retreating without crossing m=0.25. The m=1.0 eval has been **0/10
  every round** of the entire run — the model has zero capability without
  KTO mixing into the action. **This is the motivation for the SDEdit
  change**: replace the action-level blend with a denoising-noise-level
  blend so the model is forced to learn how to operate without KTO present
  in the action stream.

---

## The change to make: SDEdit-style partial denoising replaces the blend

### Current pipeline (stay aware of these seams)

```
                                    ┌─── x_mean, x_std (per-channel from training set, world units)
                                    ▼
inference()                        normalize/denormalize at the seam
  ├─ build cond (20-dim, real Box2D state + KTO guidance_q hint)
  ├─ model.predict(cond, outcome, gs=2.0)
  │     │
  │     ▼ TrainedModel.predict
  │     ├─ x_norm = sampler.sample_cfg(cond)        # 30-dim normalized
  │     │     └─ x = randn(B, 30)                    # PURE NOISE init
  │     │        for t in [100, 90, ..., 10]:
  │     │            ε_cond / ε_uncond / CFG mix
  │     │            DDIM step (eta=0)
  │     │        return x_clean (normalized)
  │     ├─ x_raw = x_norm * x_std + x_mean           # back to raw relative-world-units
  │     ├─ reshape (10, 3)
  │     └─ cps[0] = (0,0,0)                          # pin first CP
  ├─ _make_position_spline(cps, action_horizon=1.5s) # 3 cubic B-splines
  └─ _diff_q_origin = q_now (so spline values are offsets from this anchor)

get_action(margin):
  diff_rel  = _eval_spline(_diff_splines, t_sim - _diff_t0)
  diff_world = _diff_q_origin + diff_rel
  ref       = (1 - m) * kto_q + m * diff_world      # ← THE BLEND, in raw world units
  PD tracks ref
```

**Important normalization facts**:
- Model OUTPUT space = `x_norm` ∈ ℝ³⁰, normalized so train target ≈ N(0,1)
  per channel.
- Model INTERNAL space (during DDIM) = same `x_norm`. The DDIM math operates
  there. Forward noising is `x_t = √α̅·x_0 + √(1-α̅)·ε`.
- World space (raw) = `x_raw = x_norm * x_std + x_mean`, in **relative
  meters / radians offsets from the inference-time q_now**. First CP is
  forcibly zeroed because it's the anchor.
- Training targets (`actual_tracked_cps` from `_fit_kto_window`) are KTO
  trajectories fit to 10 CPs in this same raw-relative-world space, with
  the first CP at (0,0,0) by construction. So `x_mean[0:3] ≈ 0` and
  `x_std[0:3]` is small but nonzero.
- DDIM schedule: `T=100` underlying cosine schedule, `n_steps=10` actual
  reverse passes, `timesteps = [100, 90, 80, ..., 10]`. **We are NOT
  doing 100 forward passes per inference — we already do 10.**

### Proposed pipeline

```
inference(margin):                                     ← margin is now passed in
  ├─ build cond (same 20-dim)
  │
  ├─ guidance_spline_cps = _fit_kto_window(            # 10×3 in raw relative-world units
  │       kto_plan, idx_now, action_horizon=1.5s)     # exactly the model output's shape and units
  │       # If KTO plan is exhausted past t_sim, returns the last-position pad
  │       # (today's _fit_kto_window already handles this gracefully)
  │
  ├─ mask_bit = (rng.random() < margin)                # ← reuses the existing dropout draw
  │
  ├─ if mask_bit:                                      # OPTION (a) — confirmed
  │     guidance_q_override = (0,0,0)                  # mask cond
  │     init_cps = None                                # SDEdit starts from pure noise
  │ else:
  │     guidance_q_override = None                     # keep cond
  │     init_cps = guidance_spline_cps                 # SDEdit starts from this
  │
  ├─ k = round(margin * n_steps)                       # effective denoising steps
  │     # m=0.0 → k=0 → return guidance directly, NO model call
  │     # m=0.1 → k=1 → 1 forward + 1 reverse step, near-guidance
  │     # m=0.5 → k=5 → midway
  │     # m=1.0 → k=10 → standard pure-noise sampling
  │
  ├─ cps = model.predict(cond, outcome, init_cps=init_cps, k_effective=k)
  │     # NEW signature on model.predict() / TrainedModel.predict() / _LiveModel.predict
  │
  ├─ cps[0] = (0,0,0)                                  # re-pin (always)
  ├─ _diff_splines = _make_position_spline(cps, action_horizon)
  └─ _diff_q_origin = q_now

get_action(margin):
  diff_rel  = _eval_spline(_diff_splines, t_sim - _diff_t0)
  diff_world = _diff_q_origin + diff_rel
  ref       = diff_world                               # ← NO BLEND, diffusion is sole source
  # KTO velocity/accel refs still feed PD because the model only outputs position
  PD tracks ref
```

### SDEdit math, mapped to the existing schedule

```python
T = schedule.T = 100
self.timesteps = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]   # n_steps = 10
```

Standard sampling: start at index 0 (max-noise t=100), run all 10 reverse
steps to t≈0.

Partial sampling with k effective steps:

```python
start_index = n_steps - k                  # k=0 → 10 (no work) ; k=10 → 0 (full)
t_start     = timesteps[start_index]       # k=3 → timesteps[7] = 30
α̅_start     = schedule.alpha_bar[t_start]
x_init      = sqrt(α̅_start) * init_cps_norm + sqrt(1 - α̅_start) * randn_like
# then run DDIM steps from start_index..n_steps-1 as usual
```

α̅ values for our cosine T=100 schedule at the timesteps we visit:

| k | start_idx | t_start | α̅(t_start) | √α̅ | √(1-α̅) |
|---|---|---|---|---|---|
| 0 | 10 | — | — | — | (no model call, return guidance) |
| 1 | 9 | 10 | 0.972 | 0.986 | 0.167 |
| 3 | 7 | 30 | 0.781 | 0.884 | 0.468 |
| 5 | 5 | 50 | 0.499 | 0.706 | 0.708 |
| 7 | 3 | 70 | 0.218 | 0.467 | 0.884 |
| 10 | 0 | 100 | ~5e-5 | ~0.007 | ≈1.0 |

So m=0.1 (k=1): x_init ≈ 0.986·guidance + 0.167·noise — barely perturbed.
m=1.0 (k=10): x_init ≈ pure noise → standard sampling, guidance ignored.

### The first-CP pin
- `init_cps[0] = (0,0,0)` by construction (guidance is relative to q_now).
- Forward noise propagates noise into all CPs including index 0.
- After DDIM denoise, `cps[0]` may be slightly off zero due to denoising
  error.
- Re-pin `cps[0] = (0,0,0)` post-sampling for safety. This is what the
  current code does and it should stay.

### Why training is unchanged
- The model is trained with random t in [1, T] for diffusion loss — it
  already knows how to denoise from any noise level.
- SDEdit just changes how we initialize x at sampling time.
- The training-target CPs (`actual_tracked_cps` from `_fit_kto_window`)
  are the same kind of object as the proposed `init_cps` (KTO ref fit to
  10 CPs). So no train/test mismatch.
- Slight subtlety: at training time, x_noisy is built from the *target*
  ground-truth CPs; at SDEdit inference, x_init is built from the *current*
  KTO ref. These are usually nearly identical (KTO IS the target most of
  the time), so the model should generalize fine.

---

## Decisions (resolved by user)

### Q0 — denoising steps: KEEP THE EXISTING SCHEDULE
**User confirmed**: "we are ok with the existing noise schedule".
- T = 100 (cosine), n_steps = 10 DDIM passes per inference.
- timesteps = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
- No retraining needed. Model is reused as-is.
- m → k mapping: `k_effective = round(m * 10)`, so:
  - m=0.0 → k=0  → return guidance directly (skip model)
  - m=0.1 → k=1  → 1 forward + 1 reverse step (~17% noise added to guidance)
  - m=0.5 → k=5  → ~71% noise
  - m=1.0 → k=10 → standard pure-noise sampling

### Q6 — RESUME FROM `advantage_round40.pt` (mid-run cutover)
**User confirmed**: "We will resume from ckpt40".
- Use `../phase4_results/advantage_round40.pt` as the starting checkpoint.
- The trained diffusion model state is preserved.
- The advantage_archive.db is preserved (currently 50k+ frames).
- The critic 640fdf61-372 + live retargeting state restarts from frozen target.
- **Important**: starting margin must be set carefully because the SDEdit
  mechanism re-interprets margin. At the time of resume, the run was
  oscillating around m≈0.18. Under SDEdit, m=0.18 means k=2 effective steps
  (~46% noise). The model has never been trained at this specific noise
  level / regime combo, so the first few rounds will be a hard regime change.
  **Suggest starting at `--initial-margin 0.10`** (k=1, ~17% noise — close
  to KTO pass-through) to give the model room to relearn what each m means.
  Let the geometric advance push it back up.

---

## Implementation plan — exact code changes

### File 1: `repo/model.py` — `DDIMSampler` gets `sample_partial`

Add a new method below the existing `sample_cfg`. Refactor `sample_cfg` to
delegate to it (preserves backwards compat for any caller using the old
signature).

```python
@torch.no_grad()
def sample_partial(
    self,
    cond: torch.Tensor,
    init_x_norm: torch.Tensor | None = None,
    k_effective: int | None = None,
    guidance_scale: float = 2.0,
    device: str = "cpu",
) -> torch.Tensor:
    """SDEdit-style partial denoising.

    Args:
        cond:        (B, COND_DIM) conditioning
        init_x_norm: (B, X_DIM) clean signal in normalized space, or None
                     for standard pure-noise sampling
        k_effective: number of reverse DDIM steps to actually take, in
                     [0, n_steps]. None defaults to n_steps (full sampling).
                     k=0 with init_x_norm not None is a no-op return.
        guidance_scale: CFG weight (unchanged from sample_cfg)
    """
    B = cond.shape[0]
    x_dim = self.model.output_proj.out_features
    n_steps = self.n_steps
    if k_effective is None:
        k_effective = n_steps
    k_effective = max(0, min(n_steps, int(k_effective)))

    # k=0 short-circuit: if init provided, return it; else return pure noise
    if k_effective == 0:
        if init_x_norm is not None:
            return init_x_norm
        return torch.randn(B, x_dim, device=device)

    cond_uncond = cond.clone()
    cond_uncond[:, CFG_START:CFG_END] = 0.0

    # Determine where in the timesteps array to start
    start_index = n_steps - k_effective
    t_start = int(self.timesteps[start_index])

    if init_x_norm is None:
        # Standard pure-noise initialization (any k_effective)
        x = torch.randn(B, x_dim, device=device)
    else:
        # SDEdit forward noise to t_start
        alpha_bar_start = torch.tensor(
            self.schedule.get_alpha_bar(t_start), device=device
        )
        noise = torch.randn(B, x_dim, device=device)
        x = (
            torch.sqrt(alpha_bar_start) * init_x_norm
            + torch.sqrt(1.0 - alpha_bar_start) * noise
        )

    # Run DDIM steps from start_index..n_steps-1
    for i in range(start_index, n_steps):
        t_cur = int(self.timesteps[i])
        t_prev = int(self.timesteps[i + 1]) if i + 1 < n_steps else 0
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

        eps_cond = self.model(x, cond, t_batch)
        eps_uncond = self.model(x, cond_uncond, t_batch)
        eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        x = self._ddim_step(x, eps_pred, t_cur, t_prev, device)

    return x


@torch.no_grad()
def sample_cfg(self, cond, guidance_scale=2.0, device="cpu"):
    """Backwards-compat wrapper: full pure-noise sampling."""
    return self.sample_partial(
        cond, init_x_norm=None, k_effective=None,
        guidance_scale=guidance_scale, device=device,
    )
```

### File 2: `repo/eval.py:TrainedModel.predict` and the two `_LiveModel.predict` (in `dagger_loop.py` and `dagger_advantage.py`)

All three have the same shape. Add new optional kwargs:

```python
def predict(
    self,
    cond,
    outcome,
    guidance_scale=2.0,
    init_cps_world=None,    # NEW (10, 3) raw relative world units, or None
    k_effective=None,       # NEW int in [0, n_steps], or None
):
    full = np.zeros(COND_DIM, dtype=np.float32)
    full[:STATE_DIM] = cond[:STATE_DIM]
    full[CFG_START] = float(outcome)
    ct = torch.tensor(full, dtype=torch.float32).unsqueeze(0)

    init_x_norm = None
    if init_cps_world is not None:
        init_world = np.asarray(init_cps_world, dtype=np.float32).flatten()
        init_norm_np = (init_world - self.x_mean.numpy()) / self.x_std.numpy()
        init_x_norm = torch.tensor(init_norm_np, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        x_norm = self.sampler.sample_partial(
            ct, init_x_norm=init_x_norm, k_effective=k_effective,
            guidance_scale=guidance_scale,
        )

    x_raw = (x_norm * self.x_std + self.x_mean).squeeze(0).numpy()
    cps = x_raw.reshape(N_CPS, N_CHANNELS)
    cps[0] = [0, 0, 0]
    return cps
```

For the m=0 short-circuit (k_effective=0 with init not None), the sampler
returns `init_x_norm` unchanged → `x_raw = init_world` → `cps[0]` is
re-pinned to (0,0,0) (which it already was). So m=0 IS the guidance
pass-through automatically; no separate code path needed in the wrappers.

### File 3: `repo/diffusion_controller.py:KTODiffusionController.inference`

Pass margin in. Build guidance_spline. Roll mask bit. Call new model
signature.

The current signature is:
```python
def inference(self, guidance_q_override: Position | None = None) -> None:
```

New signature:
```python
def inference(
    self,
    guidance_margin: float = 0.0,
    rng: np.random.Generator | None = None,
) -> None:
```

The `guidance_q_override` parameter goes away — the controller now
makes the mask decision itself based on `guidance_margin` and the RNG.
The collector passes its rng down. Update all call sites
(`dagger_advantage.collect_episode_diverse`, `run_eval_episode`,
`critic.collect_diverse.run_diverse_episode`, and any others).

Logic inside `inference`:

```python
def inference(self, guidance_margin: float = 0.0, rng=None) -> None:
    if rng is None:
        rng = np.random
    uw = self.env.unwrapped
    L = uw.lander
    t_sim = uw.elapsed_s
    q_now = (L.position.x, L.position.y, L.angle)
    q_prev = self._last_inference_q if self._last_inference_q is not None else q_now
    vel = (L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity)
    pad_x, pad_y = solver.PAD_X, solver.PAD_Y
    dq = (pad_x - q_now[0], pad_y - q_now[1], 0.0 - q_now[2])
    dq_prime = (0.0 - vel[0], -0.5 - vel[1], 0.0 - vel[2])

    # Build guidance_spline from KTO ref window (1.5s, 10 CPs, raw world rel)
    # Use the same _fit_kto_window function dagger_loop uses for training targets.
    from dagger_loop import _fit_kto_window      # or move it into this module
    kto_idx_now = int(round((t_sim - self._kto_t0) / DT))
    if self._kto is not None:
        guidance_spline_cps = _fit_kto_window(
            self._kto.plan, kto_idx_now, self.action_horizon, dt=DT
        )
    else:
        guidance_spline_cps = np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
    guidance_spline_cps[0] = [0.0, 0.0, 0.0]   # pin

    # Mask bit (option a): with prob = margin, drop both guidance_q AND
    # guidance_spline.
    m = float(np.clip(guidance_margin, 0.0, 1.0))
    mask_bit = rng.random() < m

    if mask_bit:
        guidance_q = (0.0, 0.0, 0.0)
        init_cps = None
    else:
        kto_ref = self._get_kto_ref(t_sim)
        guidance_q = kto_ref["q"] if kto_ref else q_now
        init_cps = guidance_spline_cps

    # k_effective from margin
    k_effective = int(round(m * 10))   # 10 = sampler.n_steps; could read self.model.sampler.n_steps

    cond = _build_cond(
        t_obs_cmd_latency=DT,
        q_now=q_now, q_prev=q_prev,
        obstacle=ObstacleRelative(0.0, 0.0, 0.0),
        waypoint=WaypointTarget(dq=dq, dq_prime=dq_prime),
        guidance_q=guidance_q,
        action_horizon=self.action_horizon,
    )

    cps = self.model.predict(
        cond, self.outcome, guidance_scale=2.0,
        init_cps_world=init_cps, k_effective=k_effective,
    )
    cps[0] = [0.0, 0.0, 0.0]   # re-pin
    self._last_cps_norm = cps.copy()  # if PR1-style stash is used

    self._diff_splines = _make_position_spline(cps, self.action_horizon)
    self._diff_t0 = t_sim
    self._diff_q_origin = np.array(q_now)
    self._last_inference_q = q_now
```

### File 4: `repo/diffusion_controller.py:KTODiffusionController.get_action`

Drop the weighted-average blend.

```python
def get_action(self, guidance_margin: float = 0.001) -> ThrustVec:
    """SDEdit version: PD tracks the diffusion ref directly. The blend
    is gone — guidance_margin no longer affects get_action; it only
    affects inference's noise level and mask draw.
    """
    uw = self.env.unwrapped
    L = uw.lander
    t_sim = uw.elapsed_s

    kto_ref = self._get_kto_ref(t_sim)
    if kto_ref is None:
        return (0.0, 0.0)
    kto_v = kto_ref["v"]
    kto_a = kto_ref["a"]

    x, y, theta = L.position.x, L.position.y, L.angle
    vx, vy, omega = L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity

    if self._diff_splines is not None:
        t_diff = t_sim - self._diff_t0
        diff_rel = _eval_spline(self._diff_splines, t_diff, self.action_horizon)
        x_ref = self._diff_q_origin[0] + diff_rel[0]
        y_ref = self._diff_q_origin[1] + diff_rel[1]
        th_ref = self._diff_q_origin[2] + diff_rel[2]
    else:
        # No inference fired yet — fall back to KTO position
        kto_q = kto_ref["q"]
        x_ref, y_ref, th_ref = kto_q

    vx_ref, vy_ref, om_ref = kto_v
    ax_ref, ay_ref, al_ref = kto_a

    Fm, Fs = solver._tracking_step(
        x, y, theta, vx, vy, omega,
        x_ref, y_ref, th_ref, vx_ref, vy_ref, om_ref,
        ax_ref, ay_ref, al_ref, self.gains,
    )
    a_main = float(np.clip(2.0 * Fm / solver.THRUST_MAX - 1.0, -1.0, 1.0))
    a_side = float(np.clip(Fs / solver.SIDE_FORCE_MAX, -1.0, 1.0))
    return (a_main, a_side)
```

Note: `guidance_margin` is now an unused parameter in `get_action`. Keep
it for backwards compat with all the call sites that pass it; just don't
use it. Or remove it from the signature and update callers — your call.

### File 5: callers of `ctrl.inference()` need to pass margin and rng

In `dagger_advantage.collect_episode_diverse`:
```python
# OLD:
ctrl.inference(guidance_q_override=guidance_q_override)
# NEW:
ctrl.inference(guidance_margin=margin, rng=rng)
```

The old per-frame guidance_q dropout code in `collect_episode_diverse`
goes away — the controller now does it itself.

Same change in:
- `dagger_advantage.run_eval_episode` (passes `rng_eval`)
- `critic.collect_diverse.run_diverse_episode`
- `dagger_loop.collect_episode` (if you want the sister loop to also
  benefit, otherwise leave it alone — it's not used by phase 4)

### File 6: imports / `_fit_kto_window` location

`_fit_kto_window` currently lives in `dagger_loop.py`. Move it to a
shared location (e.g., add to `diffusion_controller.py` as a private
helper, or to a new `kto_helpers.py`) so `KTODiffusionController.inference`
can use it without importing dagger_loop. The function body is identical.

`dagger_advantage.py` already has its own copy too — consolidate.

---

## Verification & smoke test

### Unit-level checks (5 minutes)

1. **k=0 returns guidance unchanged**:
   ```python
   model = DiffusionMLP(); sampler = DDIMSampler(model, CosineSchedule())
   init = torch.randn(1, 30) * 0.1
   out = sampler.sample_partial(torch.zeros(1, 21), init_x_norm=init, k_effective=0)
   assert torch.allclose(out, init)
   ```

2. **k=n_steps with init=None matches sample_cfg**:
   ```python
   torch.manual_seed(0)
   a = sampler.sample_cfg(torch.zeros(1, 21))
   torch.manual_seed(0)
   b = sampler.sample_partial(torch.zeros(1, 21), init_x_norm=None, k_effective=10)
   assert torch.allclose(a, b)
   ```

3. **k=1 with low-noise init produces something close to init**:
   ```python
   init = torch.zeros(1, 30)
   out = sampler.sample_partial(torch.zeros(1, 21), init_x_norm=init, k_effective=1)
   # Output should be roughly init plus a small perturbation
   assert (out - init).abs().mean() < 0.5
   ```

### Integration smoke test (~2 minutes)

Run a tiny dagger_advantage with the new mechanism:

```bash
cd /Users/canopy4/habitat3/scratchpad/critic/repo
rm -rf /tmp/sdedit_smoke
/Users/canopy4/habitat3/.venv/bin/python -u -m dagger_advantage \
  --critic-db ../phase2_results/diverse_rollouts_v2.db \
  --critic-uid 640fdf61-372 \
  --rounds 2 \
  --initial-margin 0.1 \
  --margin-mode geometric \
  --target-landed 4 --target-failed 2 \
  --train-frames 30 --train-epochs 5 --batch-size 16 \
  --hidden 256 --n-blocks 6 \
  --starting-checkpoint single_strong_r10.pt \
  --smart-sampling \
  --retarget-mode polyak --retarget-update-interval 1 --retarget-tau 0.05 \
  --retarget-recent-episodes 50 --retarget-epochs 5 \
  --run-dir /tmp/sdedit_smoke 2>&1
```

Expected:
- Smart sampler still works (logs frames pulled)
- Episodes still complete
- Critic_quality print still appears
- Live critic training still happens
- Margin advances or retreats as before

If the smoke test passes, kick off the real run.

### Real run config (resume from advantage_round40.pt — Q6 confirmed)

```bash
/Users/canopy4/habitat3/.venv/bin/python -u -m dagger_advantage \
  --resume \
  --resume-round 40 \
  --starting-checkpoint ../phase4_results/advantage_round40.pt \
  --critic-db ../phase2_results/diverse_rollouts_v2.db \
  --critic-uid 640fdf61-372 \
  --rounds 60 \
  --initial-margin 0.10 \
  --margin-mode geometric --margin-advance-factor 1.2 \
    --margin-retreat-factor 0.8 --margin-floor 0.01 \
  --target-landed 40 --target-failed 10 \
  --train-frames 500 --train-epochs 200 --batch-size 32 --lr 5e-5 \
  --hidden 256 --n-blocks 6 \
  --smart-sampling \
  --sample-recent-frac 0.20 --sample-best-frac 0.40 \
    --sample-edge-frac 0.10 --sample-random-frac 0.30 \
    --recent-window-episodes 100 --best-top-k-episodes 200 \
  --alarm-v-mae 0.20 --alarm-sep 0.0 --alarm-rounds 2 \
  --retarget-mode polyak --retarget-update-interval 1 --retarget-tau 0.05 \
  --retarget-recent-episodes 200 --retarget-epochs 20 \
  --run-dir ../phase4_results --seed-offset 1700000 \
  2>&1 | tee -a ../phase4_results/run_log.txt
```

Notes:
- `--initial-margin 0.10` (not 0.18 where the run left off): under the
  new SDEdit mechanism m=0.10 means k=1 — barely perturbed guidance.
  The user expects this to give >50% landings out of the gate. If it
  doesn't, something is wrong with the SDEdit math; investigate before
  letting the run continue.
- `--resume-round 40 --rounds 60` → run reaches round 100 total.
- Same `--run-dir ../phase4_results` so margin_log.json appends to the
  existing one and the sprite plots show both eras (blend before R40,
  SDEdit after).
- Update the cron with the new task ID after launch.

---

## Things to NOT break

- The frozen target critic (`640fdf61-372`) is still used by the dagger
  loop for a_bin labeling. Don't accidentally re-use the live critic for
  that — the stability point of retargeting depends on the target being
  the labeler.
- The diffusion model's `x_mean`/`x_std` come from the checkpoint OR are
  computed from the current archive's targets in `train_on_archive`. With
  SDEdit you also use them to normalize `init_cps_world` before passing to
  the sampler. If they're stale or zeroed, SDEdit will produce garbage.
- The first-CP pin must remain. The sister controller does it; the SDEdit
  version must do it after the model output AND ensure init's first CP is
  zero too (it already is, from `_fit_kto_window` returning relative
  positions).
- The training loop is unchanged. Don't accidentally rewire training to
  use init_cps anywhere — training is pure noise prediction on noisy
  targets, same as today.
- Phase 4 is in flight. Don't kill it until you have the new code ready
  to resume from a saved checkpoint, OR until the user OKs killing.
- The cron `e14b4fec` is firing every 5 minutes. After the cutover, update
  it to point at the new task ID (delete + recreate, or modify the prompt
  in place).

---

## Quick file index

- `/Users/canopy4/habitat3/scratchpad/critic/repo/diffusion_controller.py`
  — controller, `inference`, `get_action`, `_fit_kto_window` is NOT here
  yet (move it here)
- `/Users/canopy4/habitat3/scratchpad/critic/repo/model.py`
  — `DDIMSampler`, `CosineSchedule`, `DiffusionMLP`
- `/Users/canopy4/habitat3/scratchpad/critic/repo/eval.py`
  — `TrainedModel.predict`
- `/Users/canopy4/habitat3/scratchpad/critic/repo/dagger_advantage.py`
  — `_LiveModel.predict`, `collect_episode_diverse`, `run_eval_episode`,
  `train_on_archive`, `record_round_critic_quality`, `train_live_critic`,
  the round loop in `main()`
- `/Users/canopy4/habitat3/scratchpad/critic/repo/dagger_loop.py`
  — sister loop, has `_fit_kto_window` and the old `_LiveModel`. Used as
  reference, not in active phase 4.
- `/Users/canopy4/habitat3/scratchpad/critic/repo/critic/collect_diverse.py`
  — Phase 2 collector. Has its own `run_diverse_episode`.
- `/Users/canopy4/habitat3/scratchpad/critic/SPEC.md` — the original
  experiment spec (before SDEdit was conceived)
- `/Users/canopy4/habitat3/scratchpad/critic/SPEC_addendum.md` — Phase 3.5
  + Phase 4 design changes (also pre-SDEdit)
- `/Users/canopy4/habitat3/scratchpad/critic/PLAN.md` — this file

---

## Kickoff checklist for the next session

When you (next-session-Claude) read this, do this in order:

1. Read this file end to end.
2. **Quickly check phase 4 is still alive**: `pgrep -lf "python.*dagger_advantage" | head -3` and tail the active task file (still running as `bo7oskvxu` unless something crashed).
3. **Q0 and Q6 are RESOLVED — see "Decisions" section above.**
   - Q0: keep T=100 / n_steps=10. No retraining.
   - Q6: resume from `advantage_round40.pt` with `--initial-margin 0.10`.
4. Implement the diff list above. Order: model.py first (sample_partial),
   then the predict() wrappers, then diffusion_controller.py, then the
   collector callsites, then move `_fit_kto_window`. Verify after each
   step that `import dagger_advantage` still succeeds.
5. Run the unit-level checks. Then the integration smoke test.
6. If all green, kill the live phase 4 (after a clean checkpoint save —
   round40.pt already exists, but if you're after round 45 by then, wait
   for round45.pt or use round40.pt anyway). Commit the changes. Start
   the new run with the resume config above. Update the cron with the
   new task ID. The sprite plots will keep auto-pushing (they pull from
   the same archive_db).
7. **Verify the m=0.10 expectation**: the first round under SDEdit at
   m=0.10 should give >50% landings (k=1 = barely perturbed guidance).
   If it doesn't, STOP and investigate the normalization / sampler
   plumbing. Likely culprit: x_mean/x_std mismatch when normalizing
   `init_cps_world` inside the predict() wrapper.
8. Report back to the user with: code committed, smoke green, new run
   launched, ETA, link to sprite.
