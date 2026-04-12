# C2-Continuous B-Spline Welding: Extending a 10-CP Cubic B-Spline

## Problem Statement

We have a **source** clamped cubic B-spline defined by 10 control points. At the end of this spline (or at any evaluation point), we want to create a **continuation** spline — also 10 control points — such that the junction is **C2-continuous**: position, first derivative, and second derivative all match exactly.

This arises in receding-horizon control: a planner outputs a 10-CP trajectory spline every inference step. The new prediction must seamlessly continue from the old one without discontinuities in position, velocity, or acceleration. Any jump in these quantities creates force spikes in downstream PD tracking.

## Background: Clamped Cubic B-Spline Geometry

A clamped cubic (degree `p=3`) B-spline with `n+1 = 10` control points `P_0, ..., P_9` requires `n + p + 2 = 14` knots:

```
T = [0, 0, 0, 0,  Δ, 2Δ, 3Δ, 4Δ, 5Δ, 6Δ,  7Δ, 7Δ, 7Δ, 7Δ]
     ╰─ clamped ─╯  ╰──── 6 interior ─────╯  ╰─ clamped ─╯
```

The valid parameter domain is `[0, 7Δ]` with 7 polynomial spans. The 4-fold knot multiplicity at each end forces the curve to **interpolate** the first and last control points.

### Endpoint Derivative Formulas

At the **left endpoint** `t = 0` of a clamped cubic B-spline with knot vector `T`:

| Order | Formula | Involves |
|-------|---------|----------|
| C0 | `S(0) = P_0` | `P_0` |
| C1 | `S'(0) = 3(P_1 - P_0) / (t_4 - t_1)` | `P_0, P_1` |
| C2 | `S''(0) = 6/(t_4 - t_2) · [(P_2 - P_1)/(t_5 - t_2) - (P_1 - P_0)/(t_4 - t_1)]` | `P_0, P_1, P_2` |

Since clamping gives `t_0 = t_1 = t_2 = t_3 = 0`, these simplify to:

```
S(0)   = P_0
S'(0)  = 3(P_1 - P_0) / t_4
S''(0) = (6 / t_4) · [(P_2 - P_1)/t_5 - (P_1 - P_0)/t_4]
```

At the **right endpoint** `t = T_max` (by symmetry of the clamped basis):

```
S(T_max)   = P_9
S'(T_max)  = 3(P_9 - P_8) / (T_max - t_9)
S''(T_max) = (6 / (T_max - t_9)) · [(P_9 - P_8)/(T_max - t_9) - (P_8 - P_7)/(T_max - t_8)]
```

**Key insight:** The first 3 control points `(P_0, P_1, P_2)` fully determine the curve's C2 behavior at the left endpoint. The last 3 `(P_7, P_8, P_9)` fully determine it at the right endpoint.

### Derivation of Endpoint Formulas

These formulas follow from the recursive derivative property of B-splines. The derivative of a degree-`p` B-spline is a degree-`(p-1)` B-spline with control points:

```
Q_i = p · (P_{i+1} - P_i) / (t_{i+p+1} - t_{i+1})
```

on a knot vector with one knot removed from each end.

**First derivative** (`S'`): degree 2 B-spline on `T' = {t_1, ..., t_{12}}`

```
Q_i = 3 · (P_{i+1} - P_i) / (t_{i+4} - t_{i+1}),  i = 0, ..., 8
```

At the clamped left end of `T'` (which starts `[0, 0, 0, ...]`):

```
S'(0) = Q_0 = 3(P_1 - P_0) / (t_4 - 0) = 3(P_1 - P_0) / t_4
```

**Second derivative** (`S''`): degree 1 B-spline on `T'' = {t_2, ..., t_{11}}`

```
R_i = 2 · (Q_{i+1} - Q_i) / (t_{i+4} - t_{i+2}),  i = 0, ..., 7
```

At the clamped left end of `T''` (starts `[0, 0, t_4, ...]`):

```
S''(0) = R_0 = 2(Q_1 - Q_0) / (t_4 - 0)
```

Expanding:

```
Q_0 = 3(P_1 - P_0) / t_4
Q_1 = 3(P_2 - P_1) / (t_5 - t_2) = 3(P_2 - P_1) / t_5

S''(0) = (2/t_4) · [3(P_2 - P_1)/t_5 - 3(P_1 - P_0)/t_4]
       = (6/t_4) · [(P_2 - P_1)/t_5 - (P_1 - P_0)/t_4]
```

## The Welding Solution

### Setup

Given the **source spline** `S_old` (10 CPs, known knot vector), we evaluate at the weld parameter `t_w` (typically `T_max` of the old spline) to obtain:

```
pos = S_old(t_w)       # position vector
vel = S_old'(t_w)      # velocity vector
acc = S_old''(t_w)     # acceleration vector
```

We construct a **new spline** `S_new` (10 CPs, clamped at `t = 0`) whose knot vector uses spacing `Δ_new` (which may differ from the old spline's spacing):

```
T_new = [0, 0, 0, 0,  Δ, 2Δ, 3Δ, 4Δ, 5Δ, 6Δ,  7Δ, 7Δ, 7Δ, 7Δ]
```

### Solving for Constrained Control Points

Setting `S_new(0) = pos`, `S_new'(0) = vel`, `S_new''(0) = acc` and solving:

**C0 — Position:**

```
P_0 = pos
```

**C1 — Velocity:**

```
3(P_1 - P_0) / Δ = vel
P_1 = P_0 + vel · Δ / 3
```

**C2 — Acceleration:**

```
(6/Δ) · [(P_2 - P_1)/(2Δ) - (P_1 - P_0)/Δ] = acc
```

Substituting `(P_1 - P_0) = vel·Δ/3`:

```
(P_2 - P_1)/(2Δ) = acc·Δ/6 + vel/3
P_2 = P_1 + 2Δ · (acc·Δ/6 + vel/3)
P_2 = P_1 + acc·Δ²/3 + 2·vel·Δ/3
```

Expanding fully in terms of the weld state:

```
P_0 = pos
P_1 = pos + vel·Δ/3
P_2 = pos + vel·Δ + acc·Δ²/3
```

### Summary

| CP | Formula | Role |
|----|---------|------|
| `P_0` | `pos` | Matches position |
| `P_1` | `pos + vel·Δ/3` | Matches velocity |
| `P_2` | `pos + vel·Δ + acc·Δ²/3` | Matches acceleration |
| `P_3 ... P_9` | Free (model output) | Shape the trajectory |

The 3 constrained CPs consume **no model capacity** — they are deterministic functions of the weld state and knot spacing. The model predicts only the 7 free CPs.

### Latency Analysis: The Clamped Multiplicity Advantage

A critical property of the clamped point weld: the 4-fold knot multiplicity at `t=0` compresses all 3 constrained CPs' basis functions to start at the **same parameter**. They don't occupy separate spans. The Bernstein form of the first span (`u = t/Δ ∈ [0,1]`) shows how quickly control transfers to the model:

```
u    B_0(P_0)  B_1(P_1)  B_2(P_2)  B_3(P_3=FREE)  constrained%
0.00  1.000     0.000     0.000     0.000            100%
0.25  0.422     0.422     0.141     0.016             98%
0.50  0.125     0.375     0.375     0.125             88%
0.75  0.016     0.141     0.422     0.422             58%
1.00  0.000     0.000     0.000     1.000              0%
```

At `t = Δ`, the curve **interpolates P_3** — the first free CP. The constrained CPs have fully discharged their position influence by the end of one span. The model has 100% position control at `t = Δ`.

The constrained CPs do linger past `Δ` through velocity and acceleration influence (B_1's support extends to `2Δ`, B_2's to `3Δ`), but their weight drops rapidly in higher spans. The clamped multiplicity front-loads all the constraint influence into the smallest possible region.

This means the point weld's effective latency to full position control is **1 span**, which is the theoretical minimum for C2-constrained cubic B-splines — you cannot do better because C2 requires matching 3 values (pos, vel, acc) that inevitably influence the first span's shape.

### Analogy to Taylor Expansion

Note the structural similarity to a second-order Taylor expansion at the junction:

```
f(t) ≈ pos + vel·t + (1/2)·acc·t²
```

Evaluated at `t = Δ/3, Δ`:

```
P_1 ≈ pos + vel·(Δ/3) + (1/2)·acc·(Δ/3)² ≈ pos + vel·Δ/3    (acc term is O(Δ²), small)
P_2 ≈ pos + vel·Δ + (1/2)·acc·Δ²
```

The B-spline formula gives `acc·Δ²/3` where Taylor gives `acc·Δ²/2`. The factor difference (`1/3` vs `1/2`) arises from the B-spline basis averaging — control points are not on-curve points, so their "positions" encode the curve shape through the basis functions, not through direct interpolation.

## Non-Uniform Knot Spacing

If the new spline has non-uniform knot spacing, define `h_1 = t_4` (first interior knot) and `h_2 = t_5` (second interior knot). The formulas generalize to:

```
P_0 = pos
P_1 = pos + vel · h_1 / 3
P_2 = P_1 + h_2 · (acc · h_1 / 6 + vel / 3)
```

This collapses to the uniform case when `h_1 = Δ, h_2 = 2Δ`.

## Evaluating the Source Spline at the Weld Point

To extract `(pos, vel, acc)` from the source spline at its right endpoint:

```
pos = P_9^old
vel = 3(P_9^old - P_8^old) / (T_max - t_9^old)
acc = (6/(T_max - t_9^old)) · [(P_9^old - P_8^old)/(T_max - t_9^old) - (P_8^old - P_7^old)/(T_max - t_8^old)]
```

Or, more robustly, use `scipy.interpolate.BSpline` to evaluate derivatives at any parameter value — this handles non-uniform knots and mid-spline weld points automatically.

## Implementation Architecture

The recommended function signature separates concerns:

```
welded_cps = weld_bspline_c2(
    old_cps,        # (10, D) source control points
    old_knots,      # (14,)  source knot vector
    free_cps,       # (7, D) model-predicted free control points
    new_knot_spacing,  # Δ for the new spline
    weld_param=None,   # where on old spline to weld (default: right end)
)
# Returns: (10, D) welded control points [P_0, P_1, P_2, free_0, ..., free_6]
```

The function:
1. Evaluates the source spline to get `(pos, vel, acc)` at the weld point
2. Computes `P_0, P_1, P_2` using the closed-form equations above
3. Concatenates `[P_0, P_1, P_2] + free_cps` to produce the 10-CP result

### Differentiability

All operations (spline evaluation, the three linear equations, concatenation) are differentiable. This means the weld can be used inside a neural network's forward pass — gradients flow through the free CPs and (if needed) through the old CPs and knot spacing.

### Numerical Stability

The formulas involve `Δ` and `Δ²` terms. For very small `Δ` (< 0.01s), the constrained CPs cluster tightly near `pos`, and the free CPs must do all the shaping work in a narrow region — this can cause oscillation. For typical control horizons (`Δ ≈ 0.1–0.5s`), the weld region spans a physically meaningful distance and the formulas are well-conditioned.

## Verification

A correct C2 weld satisfies three testable properties at the junction:

1. **Position**: `|S_new(0) - S_old(t_w)| < ε`
2. **Velocity**: `|S_new'(0) - S_old'(t_w)| < ε`
3. **Acceleration**: `|S_new''(0) - S_old''(t_w)| < ε`

where `ε` is machine precision (~1e-12 for float64, ~1e-5 for float32). These should be verified numerically using `scipy.interpolate.BSpline` evaluation, not by re-deriving from CPs, to guard against algebraic errors in the implementation.

---

## Part 2: Overlap Welding

### Handling Inference Delay: Look-Ahead Weld

The point weld is already optimal for latency (1 span to full position control, see above). Inference delay is handled without overlap — simply **look ahead** on the old spline:

1. At time `T_now`, the system is following the old spline at parameter `t_now`
2. Estimate inference duration `t_inf`
3. Evaluate the old spline at `t_now + t_inf` to get the future weld state
4. Model predicts free CPs conditioned on this looked-ahead state
5. By the time inference completes, the system arrives at `t_now + t_inf` and the new spline takes over with C2

```python
weld_bspline_c2(
    old_cps, old_knots, free_cps, new_knot_spacing,
    weld_param=t_now + t_inference,  # look ahead on old spline
)
```

The closed-form formulas are unchanged — they don't care where on the old spline the weld state is extracted from. This gives 7 free CPs, no span matching, and no timing assumptions baked into the knot structure. The only requirement is a reasonable estimate of `t_inference`.

### Overlap Weld (Alternative for Unknown Delay)

When inference duration is unpredictable or the look-ahead approach isn't feasible, overlap welding provides an alternative. The overlap spline replays the old trajectory for a fixed number of spans, guaranteeing C2 at the switchover regardless of exact timing.

The point weld with look-ahead is preferred when possible. The overlap is a fallback.

The overlap weld enforces C2 at the instant the model starts computing. But computation takes time. By the time the new spline is ready, the system has advanced along the old spline:

```
Time:    T (model starts)          T+Δ (prediction arrives)
         |←── inference delay ──→|
Old:     =============================>
New:     P0,P1,P2 | P3 ... P9 ──────>
         C2 here ↑  ↑ first free CP
```

At `T+Δ`, we need to switch from old to new. But C2 was enforced at `T`, not at `T+Δ`. Options:

1. **Re-weld at `T+Δ`** — compute new constrained CPs from old spline's state at `T+Δ`. Invalidates the model's prediction (it trained for a weld at `T`).
2. **Splice without C2** — switch at `T+Δ` and accept the discontinuity.
3. **Overlap weld** — design the new spline so its first span(s) replay the old spline's trajectory during the inference delay. C2 at the switchover is structural.

### 1-Span Overlap: Bézier Copying

**Core idea:** The first span of the new spline exactly reproduces the last span of the old spline. The switchover happens at the boundary between spans, where B-spline continuity guarantees C2 automatically.

For a clamped cubic B-spline, the **first span** `[0, Δ]` and the **last span** `[6Δ, 7Δ]` are both single Bézier cubics. The Bézier extraction matrix is the identity at clamped endpoints — the B-spline control points ARE the Bézier control points in these spans.

**Old spline's last span** `[6Δ, 7Δ]` with local parameter `u = (t - 6Δ)/Δ ∈ [0,1]`:

```
S_old(t) = P_6^old·(1-u)³ + P_7^old·3u(1-u)² + P_8^old·3u²(1-u) + P_9^old·u³
```

**New spline's first span** `[0, Δ]` with `u = t/Δ`:

```
S_new(t) = P_0·(1-u)³ + P_1·3u(1-u)² + P_2·3u²(1-u) + P_3·u³
```

Setting them equal:

```
P_0 = P_6^old
P_1 = P_7^old
P_2 = P_8^old
P_3 = P_9^old
```

That's it — copy the last 4 CPs of the old spline to the first 4 of the new.

**4 constrained CPs, 6 free CPs** (P_4 through P_9).

### Why C2 Holds at the Switchover

At `t = Δ` (the switchover), the new spline transitions from span 1 to span 2. This is an **interior knot** of the new spline with single multiplicity, so the B-spline basis guarantees C2 continuity across it.

Since span 1 exactly reproduces the old spline's last span:

```
S_new(Δ⁻)   = S_old(7Δ)     →  C0 ✓
S_new'(Δ⁻)  = S_old'(7Δ)    →  C1 ✓
S_new''(Δ⁻) = S_old''(7Δ)   →  C2 ✓
```

And by B-spline interior knot continuity:

```
S_new(Δ⁺) = S_new(Δ⁻)       →  C0 across switchover ✓
S_new'(Δ⁺) = S_new'(Δ⁻)     →  C1 across switchover ✓
S_new''(Δ⁺) = S_new''(Δ⁻)   →  C2 across switchover ✓
```

No derivative formulas, no numerical solve — the C2 is **structural**.

### Control Latency with 1-Span Overlap

The timeline with overlap:

```
Time:    T-Δ                  T (switchover)         T+6Δ
         |←── overlap ──→|←── free region ──────────→|
New:     P0  P1  P2  P3  | P4  P5  P6  P7  P8  P9
         (= old last 4)    (model output)
```

At the switchover `t = Δ`, the Bernstein form gives a crucial property:

```
B_0(Δ) = 0,  B_1(Δ) = 0,  B_2(Δ) = 0,  B_3(Δ) = 1
```

Therefore **`S_new(Δ) = P_3 = P_9^old`** — the new spline interpolates the old endpoint exactly. The constrained CPs have fully "discharged" their influence by the switchover. The free CPs begin their influence immediately after.

Compare to the point weld at the switchover:
- Point weld: `S(0) = P_0` (constrained). Free CPs have 0% weight.
- Overlap: `S(Δ) = P_3` (last constrained CP). Free CPs begin from 0% but the *constrained CPs are from the old spline* and naturally continue its trajectory.

The 1-span overlap matches the inference delay of one knot span — the new spline "replays" the old trajectory during computation, and the model's output takes effect exactly when the prediction arrives.

### Free CP Influence Rate

At the switchover boundary, the first free CP `P_4` has basis function:

```
B_4(Δ + δ) = δ³ / (6Δ³)    for small δ > 0
```

For comparison, the point weld's first free CP at the same physical moment:

```
B_3(δ) = δ³ / Δ³    (Bernstein form at clamped start)
```

The overlap's free CP rises **6× slower** per span. This is the cost of the structural C2 guarantee — at an interior knot, the cubic basis partitions weight across more functions than at a clamped endpoint.

However, this comparison is misleading for the receding-horizon use case. In the point weld scenario, you'd need to **re-weld at the switchover point** (computing new constrained CPs from the old spline's state at that moment), which invalidates the model's original prediction. The overlap avoids this entirely — the model trains knowing the overlap structure, and its free CPs are calibrated to the actual switchover dynamics.

### 2-Span Overlap

For longer inference delays (~2Δ), extend the overlap to 2 spans. This constrains 5 CPs, leaving 5 free.

The complication: only the first and last spans of a clamped B-spline have identity Bézier extraction matrices. The second-to-last span of the old spline is an interior span — its polynomial is a non-trivial linear combination of CPs `P_5, P_6, P_7, P_8`, weighted by non-Bernstein basis functions.

To match the second span of the new spline to the second-to-last span of the old, we solve a **collocation system**:

1. Sample the old spline at 5 points in the overlap region `[T - 2Δ, T]`
2. Build the collocation matrix `A[j,i] = N_i(t_j)` for `i = 0,...,4`
3. Solve `A · P = S_old(t_j)` for `P_0, ..., P_4`

The 5×5 system is uniquely solvable because 5 B-spline basis functions span the space of piecewise cubics over 2 spans with C2 interior continuity (5 DOF).

**Good sample points:** the Greville abscissae of the first 5 basis functions, or simply 5 uniformly-spaced points in `(0, 2Δ)` (excluding endpoints to avoid boundary artifacts).

```
Constrained: P_0, P_1, P_2, P_3, P_4  (5 CPs, from collocation)
Free:        P_5, P_6, P_7, P_8, P_9  (5 CPs, model output)
```

The collocation is a one-time solve per inference step (5×5 linear system per spatial dimension). It produces the exact overlap to machine precision.

### Comparison of Welding Strategies

| Strategy | Constrained | Free | C2 guarantee | Inference delay absorbed | Complexity |
|----------|-------------|------|-------------|--------------------------|------------|
| Point weld | 3 | 7 | At junction (closed-form) | 0 | Trivial |
| 1-span overlap | 4 | 6 | At switchover (structural) | 1Δ | Copy 4 CPs |
| 2-span overlap | 5 | 5 | At switchover (structural) | 2Δ | 5×5 linear solve |

**Recommendation:** If you can apply the new spline immediately (weld at the current state), use the **point weld** — it's simpler, has 7 free CPs, and reaches full model control in 1 span. Use overlap only when inference delay forces a timing gap between when the weld state is captured and when the spline is applied.

### Knot Spacing Choice

The overlap design creates a coupling between **knot spacing** and **inference budget**:

```
overlap_spans = ceil(inference_time / Δ)
```

If `Δ = 0.2s` and inference takes `0.15s`, a 1-span overlap suffices. If inference takes `0.35s`, you need 2-span overlap (5 constrained, 5 free). This pushes toward larger `Δ` (fewer, wider spans) to keep the overlap small, at the cost of coarser trajectory resolution.

Alternatively, use a **non-uniform knot vector** where the first span(s) are sized to the inference delay and the remaining spans are sized for trajectory resolution.
