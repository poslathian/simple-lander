# Fix Plan

Based on test failures and confirmed issues from issues.md.

## Fix 1: _tracking_step inverse dynamics — add cos(2θ) denominator

**File:** `solver.py:829`

**Current:**
```python
Fm = MASS * (-ax_des * st + (ay_des + GRAVITY) * ct)
```

**Fix:**
```python
c2t = ct * ct - st * st  # cos(2θ)
Fm = MASS * (ax_des * st + (ay_des + GRAVITY) * ct) / c2t
```

Note: The sign also needs fixing. The current formula uses `-ax_des * st` but the full inverse dynamics uses `+ax_des * st`. Looking at the derivation:

Forward model: `ax = (-Fm*st + Fs*ct)/m`, `ay = (Fm*ct - Fs*st)/m - g`

Solving the 2x2 system via Cramer's rule:
- Multiply first by `st`: `ax*st = (-Fm*st² + Fs*ct*st)/m`
- Multiply second by `ct`: `(ay+g)*ct = (Fm*ct² - Fs*st*ct)/m`
- Add: `ax*st + (ay+g)*ct = Fm*(ct²-st²)/m = Fm*cos(2θ)/m`
- So: `Fm = m*(ax*st + (ay+g)*ct) / cos(2θ)`

The current code has `Fm = m*(-ax_des*st + (ay_des+g)*ct)` which is `m*((ay+g)*ct - ax*st)` — the sign on ax_des is wrong AND the cos(2θ) is missing.

Wait, let me re-derive. Looking at the original algebra more carefully:

`-Fm*st + Fs*ct = m*ax` ... (1)
`Fm*ct - Fs*st = m*(ay+g)` ... (2)

To eliminate Fs: multiply (1) by `st` and (2) by `ct`:
- `(1)*st: -Fm*st² + Fs*ct*st = m*ax*st`
- `(2)*ct: Fm*ct² - Fs*st*ct = m*(ay+g)*ct`
- Add: `Fm*(ct²-st²) = m*(ax*st + (ay+g)*ct)`
- `Fm = m*(ax*st + (ay+g)*ct) / (ct²-st²)`

But wait: `(1)*(-st)`: `Fm*st² - Fs*ct*st = -m*ax*st` and `(2)*ct`: `Fm*ct² - Fs*st*ct = m*(ay+g)*ct`. Adding: `Fm*(st²+ct²) - Fs*st*(ct+ct) + ... ` — no, let me redo.

OK, multiply (1) by `st`: `-Fm·st² + Fs·ct·st = m·ax·st`
Multiply (2) by `ct`: `Fm·ct² - Fs·st·ct = m·(ay+g)·ct`
Add them: `Fm·(ct² - st²) = m·(ax·st + (ay+g)·ct)`
So `Fm = m·(ax·st + (ay+g)·ct) / cos(2θ)`.

The current code: `Fm = MASS * (-ax_des * st + (ay_des + GRAVITY) * ct)`. This is equivalent to `m·((ay+g)·ct - ax·st)` with NO cos(2θ) denominator. The sign on `ax` is flipped AND the cos(2θ) is missing.

Actually wait — there's an alternative method. Multiply (1) by `-ct` and (2) by `st`:
- `Fm·st·ct - Fs·ct² = -m·ax·ct`  
- `Fm·ct·st - Fs·st² = m·(ay+g)·st`

Subtract second from first: `-Fs·(ct²-st²) = -m·ax·ct - m·(ay+g)·st`
`Fs = m·(ax·ct + (ay+g)·st) / cos(2θ)`

And back to Fm, if we don't divide by cos(2θ) but instead assume cos(2θ)≈1 (small angle):
`Fm ≈ m·(ax·st + (ay+g)·ct)`

The code has `m·(-ax·st + (ay+g)·ct)` — the ax sign is wrong.

**Actually**, let me re-check. Maybe the code is doing something subtler. At θ≈0: cos(2θ)≈1, st≈0, ct≈1, so `Fm ≈ m·(ay+g)`, which is correct (main engine fights gravity). And both formulas give the same result at θ=0. The test failure at θ=0.1 confirms the mismatch.

Let me check: `m·(-ax·st + (ay+g)·ct)` vs `m·(ax·st + (ay+g)·ct)/cos(2θ)` at θ=0.1:
- ax=2, ay=-5, g=10, m≈4.817
- `-ax·st + (ay+g)·ct = -2·sin(0.1) + 5·cos(0.1) = -0.1997 + 4.975 = 4.775`
- `Fm_track = 4.817 · 4.775 = 23.0`
- `(ax·st + (ay+g)·ct)/cos(2θ) = (2·0.0998 + 5·0.995)/cos(0.2) = (0.1997+4.975)/0.9801 = 5.28`
- `Fm_full = 4.817 · 5.28 = 25.4`

The issue is the sign: `+ax·st` vs `-ax·st`. That's the primary bug — the sign on the ax term is wrong. The cos(2θ) is secondary.

**Fix:** Change sign on ax term AND add cos(2θ) denominator.

Also need to fix Fs computation similarly.

## Fix 2: guidance_controller.py — remove or update broken imports

**File:** `guidance_controller.py`

This file imports `ActionTarget` from `diffusion_controller`, which doesn't exist in the current position-spline architecture. Since guidance_controller.py is a vestigial file from the thrust-spline era and is not imported by any active code, the cleanest fix is to delete it. Similarly, `test_diffusion_pipeline.py` references non-existent types.

**Fix:** Delete `guidance_controller.py` and `test_diffusion_pipeline.py` as they reference an obsolete API.

## Fix 3: Update issues.md — correct false positive on knot mismatch

**File:** `issues.md`

The B-spline knot mismatch issue was a false positive — both `_make_position_spline` and `_fit_kto_window` produce identical clamped knot vectors. The `internal` array starts with 0.0 which overlaps with the `np.full(DEGREE, 0.0)` padding, correctly producing DEGREE+1 repeated boundary knots.

**Fix:** Remove or annotate the knot mismatch section in issues.md.

## Non-fixes (design issues, not bugs)

The following issues from issues.md are design concerns that affect training quality but don't warrant code fixes in this pass:
- Retroactive outcome labeling in collect.py
- Per-step margin jitter in dagger_loop.py  
- DaggerDB.copy_from memory usage
- NoiseModel output scale mismatch

These should be addressed as part of the training pipeline refinement, not as bug fixes.
