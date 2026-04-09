# Minimal-conditioning diffusion transformer controller

A redesign of the simple-lander diffusion controller that replaces the
existing 131-dim hand-engineered conditioning vector + 1024-dim MLP with a
~24-dim conditioning + 64-dim transformer that proved out in this scratchpad.

The key bet: **a 10-CP B-spline of the lander's recent (x, y) trajectory
already encodes everything the policy needs about the lander's state**
(position, velocity, acceleration, recent maneuver shape). All we add on
top is "where the goal is".

---

## 1. Problem framing

The expert is the KTO trajectory solver, which produces a full 5 s position
plan from any (state, goal). We want a fast learned policy that produces
*position trajectory continuations* from a tiny conditioning vector,
callable at control rate, trainable by DAgger against the KTO expert.

Position trajectories are tracked back to thrust commands by the existing
cascaded PD tracker (`KTOController.step` in `lunar_lander.py`), so the
policy never has to think about thrust directly.

---

## 2. Inputs

| Field | Shape | Purpose |
|---|---|---|
| `past_cps`     | (10, 2) | clamped 10-CP cubic B-spline fit to the last **1.5 s** of actual lander (x, y) |
| `goal`         | (2,)    | landing-pad target (x, y) |
| `goal_v`       | (2,)    | desired touchdown velocity (zero for a soft landing) |

Total: **24 floats** (vs. 131 in the original).

Optional one-shot extras (off by default, easy to add as extra tokens):
- `pad_height` — terrain height under the goal x — 1 float
- `lander_theta` — body angle — 1 float

The 1.5 s window is the same as the toy in this repo and gives the model
enough context to read off current pos/vel/acc from the CPs directly.

### Frame normalization

1. **Translate**: subtract `past_cps[0]` from `past_cps`, `goal`. The model
   only sees the lander relative to its 1.5-s-ago position.
2. **Per-channel z-score** with stats baked into the checkpoint
   (the same `mean`/`std` mechanism `WindowPairDataset` already uses).

No rotation normalisation in v1 — lunar lander has gravity along -y so the
frame already has a natural orientation.

---

## 3. Output

| Field | Shape | Purpose |
|---|---|---|
| `future_cps` | (10, 2) | clamped 10-CP cubic B-spline of the next **1.5 s** of position trajectory, starting **0.33 s into the future** |

Same representation and same window/shift as the toy. **20 floats** out
(vs. 30 in the 15-CP original). 10 CPs is plenty for a 1.5 s position arc;
the toy's val MSE 0.003 confirms this.

The 0.33 s shift gives the controller time to compute the next plan
without falling behind real time, and gives PD tracking a 0.33 s ramp-in
window.

---

## 4. Architecture

A 64-dim diffusion transformer in the same shape as the proven toy in this
repo, with one extra **goal token**.

```
tokens (21 total):
    [c0  c1  ... c9 ][g][n0  n1  ... n9]
     past CP tokens   ^   noisy target CP tokens
                      goal token
```

- `cp_embed: Linear(2 -> 64)` — shared across past and target CPs
- `goal_embed: Linear(4 -> 64)` — for `cat([goal, goal_v])`
- Learned positional embedding `(21, 64)`
- Learned segment embedding `(3, 64)` — past / goal / target
- Sinusoidal timestep embedding -> 2-layer MLP -> 64, broadcast-added to
  every token
- 4 pre-LN transformer blocks (4 heads, MLP ratio 4)
- Final LayerNorm + `Linear(64 -> 2)` over the 10 target tokens

Total params: ~250 k (vs. 6.8 M for `DiffusionMLP(hidden=1024)`).
This is a **27× smaller** model.

### Diffusion schedule

Unchanged from the toy:
- Linear betas in `[1e-4, 0.02]`, T = 1000
- ε-prediction loss
- Deterministic DDIM sampler at 50 steps for inference (≈ 30 ms on CPU)

---

## 5. Dataset

For each KTO plan in the cache, slide a window across the trajectory:

```
for each plan p (length T):
    for t0 in range(0, T - (WIN + SHIFT), stride):
        past_window   = p[t0          : t0 + WIN]
        future_window = p[t0 + SHIFT  : t0 + SHIFT + WIN]
        past_cps      = fit_window(past_window)
        future_cps    = fit_window(future_window)
        goal          = p[-1]              # final (x, y)
        goal_v        = (0, 0)             # soft landing
        anchor        = past_cps[0]
        yield (past_cps - anchor,
               goal     - anchor,
               goal_v,
               future_cps - anchor)
```

`WIN = 76`, `SHIFT = 17`, `dt = 0.02 s`. This is the exact dataset the toy
already builds, plus one extra `(goal - anchor, goal_v)` field per pair.
On the 550-plan KTO cache that's ~41 k training pairs at stride 2 — same
as we have today.

---

## 6. Training loss

Pure DDPM ε-prediction, no auxiliary losses:

```
t       ~ Uniform{0..T-1}
noise   ~ N(0, I)
x_t     = sqrt(αbar_t) * future_cps + sqrt(1 - αbar_t) * noise
ε_pred  = model(past_cps, goal, x_t, t)
loss    = MSE(ε_pred, noise)
```

No BC head, no action-space loss. The model only ever predicts position
splines; thrust generation is deferred to PD tracking. This is a major
simplification vs. today's pipeline.

---

## 7. Controller integration

Replace `DiffusionMLP` everywhere it appears in `modal_dagger_v2.py` /
`lunar_lander.py` with a thin wrapper:

```python
class DiffusionTransformerController:
    def __init__(self, ckpt_path):
        self.model, self.norm = load_ckpt(ckpt_path)
        self.history = deque(maxlen=WIN)        # last 1.5 s of (x, y)
        self.last_plan = None                   # cached future_cps + anchor
        self.last_plan_t = -inf

    def step(self, env_state, goal_xy):
        self.history.append((env_state.x, env_state.y))

        if env_state.t - self.last_plan_t >= REPLAN_DT:        # ≈ 0.33 s
            past = np.array(self.history)
            past_cps = fit_window(past)
            anchor = past_cps[0].copy()
            past_n  = (past_cps - anchor)
            goal_n  = (goal_xy - anchor)
            future_cps_n = ddim_sample(
                self.model,
                cond=encode_cond(past_n, goal_n),
                n_steps=50,
            )[0]
            self.last_plan       = future_cps_n + anchor
            self.last_plan_t     = env_state.t
            self.last_plan_anchor = anchor

        # PD-track the cached plan (existing code path)
        return pd_track(env_state, self.last_plan, self.last_plan_t)
```

Replan rate is the same 0.33 s used in today's `KTODiffusionController`.
The `pd_track` function is the same cascaded PD tracker
`KTOController.step` already implements — we just feed it our predicted
spline instead of the KTO solver's output.

---

## 8. DAgger loop changes

The existing `modal_dagger_v2.main()` drives:

1. KTO presolve (unchanged — still needed for the expert plans)
2. Rollouts at margin `m` with the current student
3. Train the student on (cond, expert_action) pairs
4. Increase margin

The only changes:

- **(cond, target) format**: instead of `(call_cond[131], call_output[30])`
  per model call, store `(past_cps[10,2], goal[2], goal_v[2], future_cps[10,2])`.
  The expert's `future_cps` is fit from the *KTO plan* sampled at the same
  window the student would see — i.e., during a student rollout, every
  time the student replans, we look up the matching window in the cached
  KTO plan and fit a 10-CP spline to it. This is the DAgger label.
- **Storage**: 24 + 20 = 44 floats per call vs. 161 today. ~3.6× smaller.
- **Replay**: identical (sample frames, train, repeat).
- **Student update**: same MSE-on-ε loss as section 6.
- **Rollout**: use `DiffusionTransformerController` instead of
  `DiffusionMLP` + the existing thrust-plan controller.

The margin schedule, holdout seeds, KTO pool, and pass criteria all stay
the same — we're swapping the policy class, nothing else.

---

## 9. What this throws away

Honest list of features we lose by going from 131 cond dims to 24:

| Lost feature | Was used for | Mitigation |
|---|---|---|
| Obstacle list             | Avoidance              | Train + test on no-obstacle scenes only in v1 |
| Fuel state                | Fuel-aware approach    | Soft-landing target + tight margin makes this mostly moot |
| Body angle / angular vel  | Tip-over recovery      | PD tracker handles this; not in the policy's job |
| KTO intermediate plan     | Bootstrapping          | DAgger replaces this — student learns by imitation |
| Terrain heightmap         | Hill avoidance         | Goal is on the pad; PD tracker handles small terrain |

Things 24 dims are *not* missing because the spline encodes them:
position, velocity, acceleration, recent thrust direction (implicit in
trajectory curvature), recent maneuver shape.

---

## 10. Open questions

1. Should the goal also include desired arrival time, or do we just trust
   "as fast as PD tracking can follow"? (Probably the latter — the KTO
   expert already has implicit timing baked into its plans.)
2. Should we keep `goal_v = 0` fixed, or expose it so the same model can
   do flybys? (Default 0 for v1; expose later.)
3. PD-tracking gains — the existing `KTOController` gains were tuned for
   15-CP plans; need to verify they're still stable on 10-CP plans
   (probably fine, the curves are smoother).
4. Test set: hold out 10 % of plan seeds at the dataset level so val MSE
   on plans the model has never seen is reportable separately from val
   MSE on novel windows of seen plans.

---

## 11. Param/compute budget vs. status quo

|            | DiffusionMLP h=1024 | This design |
|---|---|---|
| Cond dim   | 131                 | 24          |
| Output dim | 30                  | 20          |
| Params     | 6.8 M               | ~0.25 M     |
| Train epoch (CPU, 41 k frames, bs 512) | ~2 min on M-series | ~20 s (measured here) |
| Inference (50-step DDIM, B=1) | ~5 ms | ~30 ms |
| Storage / call | 161 floats | 44 floats |

The transformer is slower per inference call (more sequential ops in 50
DDIM steps), but the controller only replans every 0.33 s, so 30 ms is
~10 % of the replan budget. Acceptable.

---

## 12. Implementation order

1. **Extend `model.py`**: add `goal_embed`, segment embedding for goal,
   21-token forward (already 95 % of the toy code).
2. **Extend `data.py`**: emit `(past_cps, goal, goal_v, future_cps)` triples;
   anchor everything by `past_cps[0]`.
3. **Retrain on the same KTO cache**, log val MSE, sample plot. Should
   match the toy results.
4. **Wire `DiffusionTransformerController`** into a *fork* of
   `lunar_lander.py` on a new branch — does not touch the
   `position-dagger-test` branch yet.
5. **Local single-episode rollout** with a fixed KTO seed: verify the
   controller actually lands.
6. **Local mini DAgger** (1 round, 50 seeds, no Modal) before going to the
   full Modal loop. This is the cheap "does the policy class converge"
   check.
7. Only then: branch off `position-dagger-test`, swap the model, run the
   full Modal DAgger loop.
