# PIH Phase 1 — Implementation Notes

Implementation and validation notes for `PackageInHoleEnv` and `PackageInHoleKTOController`.
These are inputs to any reimplementation, not just historical record.

---

## What was built

- `PackageInHoleEnv` — new `gym.Env` in `lunar_lander.py`. Lander flies to a hole in a pad,
  picks up a package (with imperfect knowledge of its geometry and mass), and returns to the
  start pad.
- `PackageInHoleKTOController` — Drake B-spline KTO planner for the full pickup/extract/return
  trajectory, operating under *assumed* (not true) package parameters.
- `PIHConfig` — dataclass holding both true and assumed package parameters plus derived
  geometry properties.
- `scripts/validate_pih.py` — four-check validation harness.
- `scripts/diag_pih.py` — single-episode diagnostic: prints plan waypoints and per-step state.
- `Dockerfile.pih-validate` — self-contained Docker image (python:3.13-slim + Drake) for
  running validation without Modal.

---

## Bug 1 — Timeout too short for PIH episodes

**Symptom:** All PIH episodes ended in `timeout` immediately after the timeout fix was applied.

**Root cause:** `PackageInHoleEnv.step()` was using the base `TIMEOUT = 10.0` constant
(inherited from `LunarLander`). KTO plans for the PIH task take 8–22 simulated seconds to
execute. Any plan longer than 10 s hit the timeout before completion.

**Fix:** Added `PIH_TIMEOUT = 30.0` constant and used it throughout `PackageInHoleEnv.step()`
for the timeout check and for computing time-remaining penalties.

**For reimplementation:** Keep PIH_TIMEOUT distinct from the base TIMEOUT. If the trajectory
optimizer changes (different number of waypoints, different dynamics), recheck that PIH_TIMEOUT
exceeds the longest feasible plan duration with margin.

---

## Bug 2 — KTO was given true geometry, not assumed

**Symptom:** Parameter sweep showed 0% extraction collision at all heights — the sweep had
no signal.

**Root cause:** `PackageInHoleKTOController.__init__` called `solve_package_in_hole` with
`package_top_y = cfg.package_top_y` (the *true* top, derived from `package_height_true`).
KTO had perfect knowledge and always planned the correct extraction height, so the package
always cleared the hole regardless of the assumed/true mismatch.

**Fix:** Added `PIHConfig.assumed_package_top_y` property
(`= PIH_PAD_Y + package_height_assumed`) and passed that to the solver instead.

**For reimplementation:** The controller must always plan under *assumed* parameters. True
parameters only appear in the environment step (collision detection, mass transfer). Any place
the controller calls the solver must use `cfg.assumed_*` properties, not `cfg.*_true`.

---

## Bug 3 — `extraction_lander_y` formula missing `hole_depth`

**Symptom:** Even with correct height assumptions and the Bug 2 fix applied, the package
bottom was always below `PIH_PAD_Y` at the extraction waypoint — so every episode produced
an extraction collision.

**Root cause:** The original formula was:
```python
PIH_PAD_Y + package_height_assumed + 0.2 + PIH_LEG_OFFSET
```
This computed package-bottom-at-extraction as:
```
PIH_PAD_Y + h_assumed + 0.2 − h_assumed − hole_depth = PIH_PAD_Y − hole_depth + 0.2
```
which is *below* PIH_PAD_Y for any hole_depth > 0.2.

**Fix:** Include `hole_depth` in the formula:
```python
PIH_PAD_Y + hole_depth + package_height_assumed + 0.2 + PIH_LEG_OFFSET
```
Now package-bottom-at-extraction = `PIH_PAD_Y + 0.2` (correctly above the pad surface).

**For reimplementation:** The extraction height formula must account for the hole depth
because the package sits *below* the pad surface. Any formula that treats the pad surface as
the package reference without adding the hole depth will have this bug.

---

## Observation — Leg position at spring equilibrium

The Box2D leg body connects to the lander via a revolute joint with local anchor
`(±LEG_AWAY/SCALE, LEG_DOWN/SCALE) = (±0.667, +0.6)` in the leg's local frame. At the
spring's equilibrium angle (~0.9 rad), the leg *center* hangs approximately:

```
lander_y − 0.6 × cos(0.9) ≈ lander_y − 0.36
```

The leg fixture half-height is `LEG_H / (2 × SCALE) ≈ 0.133`, so the leg *bottom* reaches:

```
lander_y − 0.36 − 0.08 ≈ lander_y − 0.44
```

The KTO solver uses `PIH_LEG_OFFSET = LEG_DOWN / SCALE = 0.6` as the contact height offset.
The true offset is ~0.44. This 0.16 m discrepancy means Box2D leg-package contact never
fires for a KTO plan that targets `package_top_y + 0.6`: the legs physically miss the package
top by ~0.16 m.

**Consequence:** Box2D contact-based attachment detection is unreliable for this geometry.
Proximity-based attachment (see below) is required.

**For reimplementation:** Do not rely on Box2D leg-package contact events for attachment
detection. The leg geometry makes the effective contact offset seed-dependent and spring-
angle-dependent. Use proximity detection with a tolerance that covers the discrepancy.

---

## Design decision — Proximity-based attachment

**Problem:** Legs hang 0.44 m below the lander pivot, not 0.6 m (see above). KTO plans
contact at `package_top + 0.6`. Legs never physically reach the package top from this height.

**Solution:** Attachment is detected analytically:
```python
near_x = abs(pos.x - PIH_PICKUP_X) < 1.0        # within leg span
near_y = abs(pos.y - cfg.contact_lander_y) < 0.5  # ±0.5 of true contact height
low_speed = abs(vel.y) < 1.5 and abs(vel.x) < 0.5
```
When all three conditions hold, the package Box2D body is destroyed, its mass is transferred
to the lander body (`lander.massData.mass += cfg.package_mass_true`), and `_attached = True`.

The ±0.5 m tolerance in `near_y` covers both the 0.16 m leg-offset discrepancy and normal
tracking error. The lander typically reaches its minimum y at ~0.3–0.5 m above the KTO
contact waypoint before the plan's thrust brings it back up.

---

## Design decision — Extraction collision gate at `extraction_lander_y`

**Problem (early version):** The extraction collision check was gated on
`pos.y > contact_lander_y`. This fired during the *ascent* phase, before the plan's return
arc started. Two false-positive mechanisms:
1. At the moment of attachment, floating-point arithmetic made `lateral_dev` (e.g., 0.070000...028)
   marginally exceed the threshold (0.07), triggering immediate termination.
2. After attachment, the lander continues descending ~0.4–0.8 m due to momentum (open-loop
   KTO plan mass mismatch before vs. after attachment). Lateral drift during this descent
   triggered false collisions.
3. Lateral tracking drift (~7 cm) accumulated during the ~1 m ascent from contact to
   package-clearing height, causing 100% extraction collision even for packages the KTO
   correctly *over*estimated (h=1.3, where true protrusion 0.3 < assumed 0.5).

**Fix:** Gate on `pos.y >= cfg.extraction_lander_y` instead. This only fires once the
plan's designated extraction waypoint is reached. The semantics are: "did the package fail
to clear the hole by the time the plan declared extraction complete?"

**Consequence on sweep:** The collision threshold becomes deterministic from geometry:
collision fires when `h_true > hole_depth + h_assumed + 0.2`. With hole_depth=1.0 and
h_assumed=0.5, this is h_true > 1.7. In the sweep [1.0, 1.3, 1.6, 2.0, 2.5]:
- h ≤ 1.6 → 0% collision (package clears before return arc)
- h ≥ 2.0 → 100% collision (package still in hole; return arc immediately exceeds threshold)

**For reimplementation:** Gate extraction collision detection on the plan's extraction
waypoint height, not the contact height. Any gate lower than `extraction_lander_y` will
produce false positives from tracking drift during ascent.

---

## Validation results (2026-04-17, branch `ari/new-env`)

```
[1] Regression — heuristic on base LunarLander:  12/20 = 60%  PASS (≥40%)
[2] Parameter sweep (5 seeds per height):
      h=1.00 (prot=0.00, assumed=0.50):  0%
      h=1.30 (prot=0.30, assumed=0.50):  0%
      h=1.60 (prot=0.60, assumed=0.50):  0%
      h=2.00 (prot=1.00, assumed=0.50): 100%
      h=2.50 (prot=1.50, assumed=0.50): 100%
      Monotone increase: PASS
[3] Failure mode isolation (5 seeds each):
      depth_wrong_mass_ok  → extraction_collision: 5/5  PASS
      depth_ok_mass_wrong  → flyaway: 5/5  (mass overestimate causes over-thrust)
      both_wrong           → extraction_collision: 5/5
[4] Hover fallback: INCONCLUSIVE (solver always converged in 2–3 s)
      Note: kto_feasible flag works; extreme geometry + 2 s budget still solves.
            Reduce budget further or use more extreme geometry if hard confirmation needed.
```

---

## File map

| File | Role |
|------|------|
| `lunar_lander.py` | `PackageInHoleEnv`, `PackageInHoleKTOController`, `PIHConfig`, `_PIHContactDetector`, `PIH_TIMEOUT`, `PIH_START_X`, `PIH_PICKUP_X`, `PIH_PAD_Y`, `PIH_LEG_OFFSET` |
| `solver.py` | `solve_package_in_hole()` — Drake B-spline KTO for PIH |
| `scripts/validate_pih.py` | Four-check validation harness; saves JSONL per episode |
| `scripts/diag_pih.py` | Single-episode diagnostic with step-level state printing |
| `Dockerfile.pih-validate` | python:3.13-slim + Drake + gymnasium[box2d]; no Modal needed |
| `results/pih_validation/` | Sweep JSONL, isolation JSONL, fallback JSONL, summary JSON |
