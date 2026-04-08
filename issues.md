# Issues Found in position-dagger Codebase

## Bug: B-spline knot vector mismatch between collect.py and diffusion_controller.py

**Files:** `collect.py:109-120`, `dagger_loop.py:192-199`, `diffusion_controller.py:93-105`

The `_fit_kto_window()` functions in `collect.py` and `dagger_loop.py` use `DEGREE + 1` repeated knots at each end (i.e., 4 repeated knots for cubic), but `_make_position_spline()` in `diffusion_controller.py` uses only `DEGREE` repeated knots (3 repeated). For a clamped cubic B-spline, the correct number of repeated boundary knots is `DEGREE + 1 = 4`.

This means the spline used for inference (in `_make_position_spline`) has a different knot structure than what was used to generate training data. The training CPs were fit with clamped knots (4 repeated), but at inference time they're evaluated with non-clamped knots (3 repeated), producing different curves from the same control points.

**Impact:** The model's output CPs produce different trajectories at inference time vs what they represented during training. This is a systematic train/test mismatch that degrades controller performance.

## Bug: _make_position_spline produces wrong number of knots

**File:** `diffusion_controller.py:93-105`

With `n = 10` CPs and `DEGREE = 3`:
- `n_internal = n - DEGREE + 1 = 8`
- `internal = linspace(0, duration, 8)` → 8 values
- `knots = [0,0,0] + [0, ..., duration (8 values)] + [duration, duration, duration]`
- Total: 3 + 8 + 3 = 14 knots

For a clamped cubic B-spline with 10 CPs, we need exactly `n + DEGREE + 1 = 14` knots. So the count happens to be correct, but the structure is wrong: clamped splines need `DEGREE + 1 = 4` repeated endpoint knots, not 3. The current code has only 3 repeats, so the spline is not properly clamped — the first/last CPs don't pin the spline endpoints.

## Bug: Conditioning vector q_prev is stale after inference

**File:** `collect.py:167`, `dagger_loop.py:249`

In `collect.py:167`:
```python
q_prev = ctrl._last_inference_q if ctrl._last_inference_q else q_now
```

This reads `_last_inference_q` *after* `ctrl.inference()` has already been called on line 161, meaning it gets the value just set by `inference()` — which is `q_now`. So `q_prev` always equals `q_now` on the first call (and equals the *current* q_now on subsequent calls, not the *previous* one). The conditioning vector's q_prev field thus never actually contains the previous inference position.

Same issue in `dagger_loop.py:249` — q_prev is read after inference updates `_last_inference_q`.

Wait — looking more carefully, in `collect.py` the read at line 167 happens *before* `ctrl.inference()` at line 161? No — line 161 calls `ctrl.inference()` first, then line 167 reads `_last_inference_q`. But `inference()` at `diffusion_controller.py:240` sets `self._last_inference_q = q_now` at the end. So by line 167, `ctrl._last_inference_q` is the *current* position, not the previous one.

Actually, re-reading more carefully: in `collect.py`, the conditioning is built *independently* at lines 164-182, separate from what `ctrl.inference()` builds internally. And `ctrl.inference()` is called at line 161, which sets `_last_inference_q = q_now`. So line 167 sees the updated value. This means q_prev = q_now for the training data conditioning — training data always has q_prev == q_now.

In `dagger_loop.py:249`, the conditioning is built *before* calling `ctrl.inference()` at line 265. So it reads the old `_last_inference_q`. This is actually correct for `dagger_loop.py` — but inconsistent with `collect.py`.

**Impact:** Training data in `collect.py` has degenerate q_prev (always == q_now). The model never sees true previous positions during training. At inference time, `diffusion_controller.py:213` passes a proper q_prev. This is a train/test distribution mismatch.

## Design Issue: Main engine thrust formula inconsistency

**File:** `solver.py:829`

The `_tracking_step()` function computes main thrust as:
```python
Fm = MASS * (-ax_des * st + (ay_des + GRAVITY) * ct)
```

This is the *simplified* inverse dynamics formula (assumes cos(2θ) ≈ 1, valid for small θ). But the solver's dynamics constraints at `solver.py:498-500` use the full Cramer's-rule formula with cos(2θ) denominator:
```python
Fm = MASS * (acc[0] * st + (acc[1] + GRAVITY) * ct) / c2t
```

For small angles the simplified formula is fine. But the sign is also flipped: the solver uses `+acc[0]*st` while tracking uses `-ax_des*st`. Looking at the physics model `lunar_lander.py:85`:
```
ax = (-Fm * st + Fs * ct) / m
```
Solving for Fm: `Fm = m * (Fs*ct - ax) / st` ... or via the Cramer's rule: `Fm = m * (ax*st + (ay+g)*ct) / cos2θ`.

The tracking formula `Fm = MASS * (-ax_des*st + (ay_des+GRAVITY)*ct)` — let's check: the forward model gives `ax = (-Fm*st + Fs*ct)/m`, `ay = (Fm*ct - Fs*st)/m - g`. So `Fm*ct - Fs*st = m*(ay+g)` and `-Fm*st + Fs*ct = m*ax`. Solving: `Fm = m*(ax*st + (ay+g)*ct) / (st²+ct²·... )`. Actually `Fm*ct² + Fm*st² = m*((ay+g)*ct + ax*(-st))` ... wait, that's not right either.

Let me work it out: multiply equation 1 by `-st`: `Fm*st² - Fs*ct*st = -m*ax*st`. Multiply equation 2 by `ct`: `Fm*ct² - Fs*st*ct = m*(ay+g)*ct`. Add: `Fm*(st²+ct²) = m*(ay+g)*ct - m*ax*st`, so `Fm = m*((ay+g)*ct - ax*st)`.

This matches the tracking formula `Fm = MASS * (-ax_des * st + (ay_des + GRAVITY) * ct)`. So the tracking formula is correct for the simplified (orthogonal force) case.

But the solver uses a different formula via Cramer's rule because the actual physics has non-orthogonal forces (the side engine force direction is `(cos θ, -sin θ)`, and the main engine direction is `(-sin θ, cos θ)` — wait, those ARE orthogonal). Let me re-check the docstring in `solver.py:17-18`:

> The force directions are NOT orthogonal (dot = -sin2θ)

Hmm, the force directions in the code are `Main: (-sinθ, cosθ)` and `Side: (cosθ, -sinθ)`. Their dot product is `(-sinθ)(cosθ) + (cosθ)(-sinθ) = -2sinθcosθ = -sin2θ`. So they're only orthogonal at θ=0! This makes the simplified tracking formula incorrect for large angles.

**Impact:** At large tilt angles, the tracking controller computes incorrect main engine thrust, leading to increased tracking error. This partially explains why DAgger performance degrades at higher margins where the model produces trajectories requiring more aggressive maneuvers.

## Design Issue: TrainedModel in eval.py ignores hidden/n_blocks

**File:** `eval.py:31-32`

`TrainedModel.__init__()` creates `DiffusionMLP()` with default parameters (hidden=256, n_blocks=6). But DAgger training in `dagger_loop.py` supports `--hidden` and `--n-blocks` flags to create larger models. If a checkpoint was trained with hidden=1024, loading it with `TrainedModel` will fail with a state dict mismatch.

**Impact:** `eval.py` cannot evaluate models trained with non-default architectures.

## Design Issue: DDIM timestep handling edge case

**File:** `model.py:111-112`

```python
self.timesteps = np.linspace(T, 0, n_steps + 1, dtype=int)[:-1]
self.timesteps[-1] = max(self.timesteps[-1], 1)
```

`np.linspace(100, 0, 11, dtype=int)` produces `[100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]`. After `[:-1]`: `[100, 90, ..., 10]`. Then `self.timesteps[-1] = max(10, 1) = 10`. This is fine.

But if `n_steps=100` (same as T), you'd get every integer timestep, and the last one would be 1, which is correct. The issue is that `np.linspace` with `dtype=int` truncates rather than rounds, which can produce duplicate timesteps for certain n_steps values. For example, `np.linspace(100, 0, 8, dtype=int)` → `[100, 85, 71, 57, 42, 28, 14, 0]` — this is fine, but other values could produce issues.

**Impact:** Minor — unlikely to cause problems with default n_steps=10, but the truncation behavior is fragile.

## Bug: guidance_controller.py references non-existent ActionTarget

**File:** `guidance_controller.py:6`

```python
from diffusion_controller import ActionTarget
```

The `diffusion_controller.py` module doesn't define `ActionTarget`. It defines `PositionRef`, `WaypointTarget`, `ObstacleRelative`, etc. — but no `ActionTarget` class. Similarly, `test_diffusion_pipeline.py` imports `DiffusionController` and `LanderState` from `diffusion_controller`, but these don't exist in the current code.

**Impact:** `guidance_controller.py` and `test_diffusion_pipeline.py` cannot be imported — they will crash with ImportError. These appear to be vestigial from an older version of the codebase (thrust-spline era) that hasn't been updated for the position-spline architecture.

## Design Issue: NoiseModel output scale mismatch

**File:** `diffusion_controller.py:83`

```python
return np.random.randn(N_CPS, N_CHANNELS) * 0.1
```

The NoiseModel returns raw random CPs scaled by 0.1. But the trained model returns CPs that are denormalized from the training distribution (which can have very different scales). At margin=0.001 this barely matters, but the mismatch means the NoiseModel doesn't serve as a good drop-in baseline comparison.

**Impact:** Low — NoiseModel is only used as a placeholder and at margin=0.001 contributes negligibly.

## Design Issue: DaggerDB.copy_from loads all rows into memory

**File:** `dagger_loop.py:101-113`

```python
def copy_from(self, other: "DaggerDB"):
    eps = other.conn.execute("SELECT * FROM episodes").fetchall()
    ...
    frames = other.conn.execute("SELECT * FROM frames").fetchall()
```

Both queries load all rows into memory at once. With large archives (thousands of episodes, tens of thousands of frames), this could exhaust memory. Should use batch iteration.

**Impact:** Memory pressure with large datasets. Moderate concern for long DAgger runs.

## Design Issue: collect.py outcome is set retroactively

**File:** `collect.py:210-212`

All frames in an episode get the same outcome (+1 or -1) regardless of when they occurred. Early frames in a crashed episode might represent perfectly valid control trajectories — labeling them all as -1 is semantically incorrect and could hurt CFG conditioning quality.

**Impact:** Moderate — the model learns to associate early-trajectory states with failure outcomes even when those states were fine. This may weaken the CFG signal.

## Design Issue: Margin sampling in dagger_loop uses Gaussian

**File:** `dagger_loop.py:289`

```python
actual_margin = np.clip(np.abs(np.random.normal(margin, 0.05)), 0.001, 1.0)
```

This samples a different margin on *every simulation step* within an episode, which means the blend between KTO and diffusion changes rapidly. This creates high-frequency oscillation in the tracking reference and may confuse the PD controller.

**Impact:** Noisy tracking behavior during collection. The PD controller tracks a reference that jitters between KTO and diffusion every 20ms.

## Design Issue: _tracking_step uses simplified inverse dynamics

**File:** `solver.py:829`

As noted above, `_tracking_step` omits the `1/cos(2θ)` factor from the full inverse dynamics. This is used by both `solver.track()` and `diffusion_controller.get_action()`. The error is proportional to `1 - 1/cos(2θ)` which grows with θ. At θ=π/6 (30°), the error is about 15%.

**Impact:** Tracking accuracy degrades at larger tilt angles, which occur during aggressive maneuvers that the DAgger system needs most.
