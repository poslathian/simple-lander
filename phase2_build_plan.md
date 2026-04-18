# Phase 2: Oracle KTO + C² Weld Infrastructure
## Build Plan for Claude Code

**Prerequisite:** Phase 1 complete and tagged on `ari/new-env`. All four validation
checks pass. The PIH environment produces the coupled depth/mass failure mode
correctly.

**Goal:** Build the infrastructure that will eventually provide supervision for DP
training: an oracle KTO that has access to true package parameters, and a C² welding
function (Clip) that splices a new trajectory onto an executing one with position,
velocity, and acceleration continuity.

**Architectural principle (from AJ):** The replan mechanism is a callable service.
Any trigger can invoke it. Phase 2 implements two triggers (on-contact, every-N-seconds)
as configuration, not as hardcoded events.

---

## What Phase 2 Builds

Four components, in dependency order:

1. **Oracle KTO solver** — same solver, true parameters
2. **C² weld function** — implements the Clip mechanism from WRITEUP.md
3. **Trigger layer** — configurable replan triggers
4. **Data collection harness** — records everything for Phase 3

---

## Component 1: Oracle KTO

### What It Does

Identical to `plan_pih_with_kto()` but receives true parameters instead of assumed.
Produces a trajectory that avoids the extraction collision (because it knows the true
package depth) and uses correctly-sized return thrust (because it knows the true mass).

### Implementation

Add `oracle_plan_pih_with_kto()` to `pih_solver.py`. It should be a thin wrapper:

```python
def oracle_plan_pih_with_kto(
    current_state: np.ndarray,   # [x, y, theta, vx, vy, omega]
    cfg: PIHConfig,              # has both assumed and true parameters
    remaining_waypoints: list,   # which named waypoints remain (from plan)
) -> dict:
    """Plan from current_state using TRUE parameters.

    Returns same plan dict as plan_pih_with_kto():
      {feasible, times, plan, knot_xy, waypoints}
    """
    # Build an oracle config that substitutes true for assumed
    oracle_cfg = cfg.with_true_as_assumed()  # new method on PIHConfig
    return plan_pih_with_kto(
        start=current_state,
        cfg=oracle_cfg,
        remaining_waypoints=remaining_waypoints,
    )
```

### Required Changes

**PIHConfig:** Add a `with_true_as_assumed()` method that returns a copy where
`package_height_assumed = package_height_true` and
`package_mass_assumed = package_mass_true`. All computed properties (extraction_lander_y,
contact_lander_y, etc.) automatically update because they derive from assumed values.

**plan_pih_with_kto():** Must accept a `start` parameter (current state, not always the
initial starting pad position) and a `remaining_waypoints` parameter (which named
waypoints are still ahead). The oracle replans from the current state forward, not from
episode start. This may require refactoring the existing function if it currently
hardcodes the start position.

### What NOT to Change

Do not modify the existing `plan_pih_with_kto()` behavior when called without oracle
parameters. The non-oracle path must remain unchanged for Phase 1 regression.

### Important: Oracle Replan vs. Waypoint Replan

Phase 2's oracle replan and the future waypoint-triggered replan (Phase 3, user input)
are different operations that use the same weld infrastructure but with different
semantics. Claude Code should implement the oracle case only and not over-engineer
for the waypoint case.

**Oracle replan (this phase):** The oracle has correct parameters and produces a new
plan for the entire remaining trajectory. It plans **from the current state forward**.
The weld function splices the oracle's plan onto the executing trajectory at the
current moment. The oracle doesn't need to go back in time — it's replacing the
plan's future with a better-informed future, not editing the plan's past.

**Waypoint-triggered replan (future phase, user input):** A user provides a new
intermediate waypoint. The system goes back to the **previous waypoint** on the
existing plan and replans from there, bending toward or through the new user waypoint.
This reshapes already-committed trajectory to smoothly incorporate new intent. The
"at least 3 CPs back" discussion and the configurable span parameter in Q1 are
primarily about this case — how far back do you need to go to produce a smooth
trajectory through the new point?

The difference: oracle replan changes the **parameters** (depth, mass) while keeping
the same goal structure. Waypoint replan changes the **intent** (go through a different
point) and requires modifying trajectory that precedes the current execution point.

Both use `weld_c2()` to enforce C² continuity, but the `weld_param` has different
semantics:
- Oracle: `weld_param = t_current` (splice at where we are now)
- Waypoint: `weld_param = t_previous_waypoint` (splice further back in time)

For Phase 2, `weld_param` should always be the current execution parameter on the
old spline. Do not implement back-in-time splicing.

---

## Component 2: C² Weld Function

### Reference

The welding math is fully specified in WRITEUP.md (uploaded to this conversation).
Claude Code should read that document in full before implementing. The key formulas
are in the "Welding Solution" section.

### What It Does

Takes an executing trajectory (the "old" spline) and a new trajectory (from oracle
KTO), and produces a single spliced trajectory that:
- Is C²-continuous at the splice point (position, velocity, acceleration match)
- Follows the old trajectory up to the splice point
- Follows the new trajectory after the splice point (with constrained CPs ensuring
  smooth transition)

### Three Welding Strategies

WRITEUP.md describes three strategies. Implement all three behind a single interface
with a `strategy` parameter:

**Point weld** (`strategy="point"`):
- 3 constrained CPs (P0, P1, P2), 7 free CPs
- C² at junction via closed-form formulas
- No inference delay absorbed
- Formulas from WRITEUP.md:
  ```
  P_0 = pos
  P_1 = pos + vel * Δ / 3
  P_2 = pos + vel * Δ + acc * Δ² / 3
  P_3 ... P_9 = free (from oracle plan)
  ```

**1-span overlap** (`strategy="overlap_1"`):
- 4 constrained CPs (copy last 4 CPs of old spline), 6 free CPs
- Structural C² at switchover (interior knot continuity)
- Absorbs 1Δ inference delay
- Implementation: copy P_6, P_7, P_8, P_9 of old spline to P_0, P_1, P_2, P_3 of new

**2-span overlap** (`strategy="overlap_2"`):
- 5 constrained CPs (via collocation), 5 free CPs
- Structural C² at switchover
- Absorbs 2Δ inference delay
- Implementation: solve 5×5 collocation system per WRITEUP.md

### Interface

```python
def weld_c2(
    old_cps: np.ndarray,         # (10, D) control points of executing spline
    old_knots: np.ndarray,       # knot vector of executing spline
    new_free_cps: np.ndarray,    # free CPs from oracle plan
    new_knot_spacing: float,     # Δ for the new spline
    weld_param: float,           # parameter on old spline where weld occurs
    strategy: str = "point",     # "point", "overlap_1", "overlap_2"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (new_cps, new_knots) — the complete welded spline.

    new_cps has shape (10, D):
      - First 3/4/5 CPs are constrained (depending on strategy)
      - Remaining CPs are from new_free_cps

    The welded spline is C2-continuous with the old spline at the weld point.
    """
```

### Configurable Span Parameter

The "how far back" question (Q1 from deferred experiments) maps to the `weld_param`
argument. For point weld, `weld_param` is where on the old spline the weld state
(pos, vel, acc) is evaluated. For overlap strategies, `weld_param` determines which
span of the old spline gets copied.

Do NOT hardcode `weld_param` to any specific value. The caller (trigger layer) decides
where to weld. The Q1 experiment will sweep this parameter.

### Verification

WRITEUP.md specifies three testable properties (Section: "Verification"):

1. `|S_new(0) - S_old(t_w)| < ε` (position continuity)
2. `|S_new'(0) - S_old'(t_w)| < ε` (velocity continuity)
3. `|S_new''(0) - S_old''(t_w)| < ε` (acceleration continuity)

where `ε` is machine precision (~1e-12 for float64).

**Use `scipy.interpolate.BSpline` for verification**, not re-derivation from CPs.
This is explicitly recommended in WRITEUP.md to guard against algebraic errors.

Write a test that:
1. Creates a random old spline (10 CPs, clamped cubic)
2. Welds a new spline using each of the three strategies
3. Evaluates both splines at the weld point using scipy
4. Asserts all three properties hold within tolerance
5. Runs this for 100 random splines to catch edge cases

This test must pass before any integration with the PIH environment.

---

## Component 3: Trigger Layer

### What It Does

Connects oracle KTO + C² weld to the episode execution loop. The trigger decides
*when* to invoke the replan; the oracle + weld handle *how*.

### Two Trigger Modes

Implement both as configuration options on `PackageInHoleKTOController`:

**On-contact trigger** (`replan_trigger="on_contact"`):
- Fires once, at the moment package attachment is detected
- This is the point where proprioceptive information (mass mismatch) first becomes
  available
- Calls oracle_plan_pih_with_kto() from current state with true parameters
- Calls weld_c2() to splice the oracle plan onto the executing trajectory
- Controller switches to tracking the welded trajectory

**Periodic trigger** (`replan_trigger="periodic"`, `replan_interval_s=N`):
- Fires every N seconds of simulation time
- Each firing calls oracle + weld
- Controller switches to tracking the latest welded trajectory
- On-contact can also fire alongside periodic (they're not mutually exclusive)

**No trigger** (`replan_trigger="none"`):
- Baseline behavior. No oracle, no weld. KTO executes as planned.
- This is the Phase 1 behavior and must remain the default.

### Interface

```python
controller = PackageInHoleKTOController(
    env=env,
    cfg=cfg,
    replan_trigger="on_contact",     # or "periodic" or "none"
    replan_interval_s=2.0,           # only used if trigger="periodic"
    weld_strategy="point",           # passed to weld_c2
)
```

### What the Trigger Does NOT Do

The trigger does not decide the weld_param (how far back to splice). For Phase 2,
use a default of `t_current` (weld at the current execution point on the old spline,
i.e., point weld at the current state). The Q1 experiment will sweep this later.

---

## Component 4: Data Collection Harness

### What It Does

Runs episodes and saves everything needed for Phase 3 training. The output is a
per-episode record that captures the full trajectory provenance.

### Per-Episode Record

```python
{
    # Episode config
    "seed": int,
    "cfg": PIHConfig.to_dict(),          # all true and assumed parameters

    # Full trajectory data (every sim step)
    "states": np.ndarray,                # (T, 6) — x, y, theta, vx, vy, omega
    "actions": np.ndarray,               # (T, 2) — commanded thrust
    "measured_accel": np.ndarray,         # (T, 3) — actual acceleration from Box2D
    "timestamps": np.ndarray,            # (T,) — sim time

    # KTO plan (original, wrong assumptions)
    "kto_plan": {
        "cps": np.ndarray,               # control points
        "knots": np.ndarray,             # knot vector
        "waypoints": dict,               # named waypoints with positions and times
        "feasible": bool,
    },

    # Replan events (list — could be 0, 1, or many depending on trigger)
    "replan_events": [
        {
            "trigger": str,              # "on_contact" or "periodic"
            "sim_time": float,           # when the replan fired
            "state_at_trigger": np.ndarray,  # (6,) state when trigger fired
            "oracle_plan": {
                "cps": np.ndarray,
                "knots": np.ndarray,
                "waypoints": dict,
                "feasible": bool,
            },
            "welded_plan": {
                "cps": np.ndarray,
                "knots": np.ndarray,
                "weld_param": float,
                "strategy": str,
            },
        },
        # ... more events if periodic trigger
    ],

    # Outcome
    "termination_reason": str,           # SUCCESS, CRASH, TIMEOUT, etc.
    "total_reward": float,
    "episode_length_s": float,

    # Attachment event
    "attachment_time": float | None,     # sim time of package contact
    "attachment_state": np.ndarray | None,  # (6,) state at attachment
}
```

### Script

Create `scripts/collect_pih_phase2.py`:

```python
"""
Collect PIH episodes with oracle replan for Phase 3 training.

Usage:
  python scripts/collect_pih_phase2.py \
    --n_episodes 200 \
    --trigger on_contact \
    --weld_strategy point \
    --out results/phase2/collected.jsonl
"""
```

Save as JSONL (one JSON object per line, append-safe). Same pattern as Phase 1's
data collection — if the script crashes at episode 147, resume without re-running
earlier episodes.

---

## Validation for Phase 2

### Check 1: C² Weld Unit Tests

Run the standalone weld verification (100 random splines × 3 strategies). All three
continuity properties must hold within float64 tolerance (~1e-12).

### Check 2: Oracle Execution Success

Compare episode outcomes on a fixed 50-seed set:

| Condition | Expected |
|---|---|
| No replan (Phase 1 baseline) | ~60% success on normal seeds; extraction collision on depth-wrong seeds |
| Oracle replan on contact | Higher success rate on depth-wrong seeds — oracle knows true depth and replans to avoid extraction collision |
| Oracle replan on contact | Higher success rate on mass-wrong seeds — oracle knows true mass and plans correctly-sized return thrust |

The specific test: run the Phase 1 `both_wrong` scenario (which reliably produces
EXTRACTION_COLLISION without replan). With oracle replan on contact, these episodes
should succeed. If they don't, either the oracle plan is wrong or the weld is
introducing a discontinuity that destabilizes tracking.

### Check 3: Data Collection Integrity

Run `collect_pih_phase2.py` for 20 episodes. Verify:
- Every episode has a complete record (no missing fields)
- Replan events are present for the correct trigger type
- Oracle plan is different from KTO plan (true params ≠ assumed params)
- Welded plan's first 3 CPs satisfy the C² formulas from WRITEUP.md
- States and actions arrays have consistent lengths

### Check 4: Phase 1 Regression

Run the Phase 1 validation suite with `replan_trigger="none"`. All four Phase 1
checks must still pass. The oracle/weld code must not affect the no-replan baseline.

---

## Files Expected to Change

| File | Change |
|---|---|
| `pih_env.py` | Add `PIHConfig.with_true_as_assumed()` method. Expose attachment event state for data collection. |
| `pih_solver.py` | Add `oracle_plan_pih_with_kto()`. Refactor `plan_pih_with_kto()` to accept `start` and `remaining_waypoints` parameters. Add `PackageInHoleKTOController` replan trigger config and weld integration. |
| `pih_weld.py` | **NEW** — C² weld function implementing all three strategies from WRITEUP.md. Pure math, no env or solver dependencies. |
| `scripts/collect_pih_phase2.py` | **NEW** — data collection harness. |
| `scripts/test_weld.py` | **NEW** — standalone weld verification (100 random splines × 3 strategies). |
| `scripts/validate_pih.py` | Update to support replan trigger options for Check 2. |

### Files NOT Changed

| File | Reason |
|---|---|
| `kto_lander.py` | Already has what we need (feedback tracker, control function) |
| `lunar_lander.py` | Not part of PIH |
| `model.py` | No DP changes in Phase 2 |
| `diffusion_controller.py` | No DP changes in Phase 2 |

---

## What NOT to Do in Phase 2

- Do not train DP. That's Phase 3.
- Do not implement operator UI or interactive waypoint selection.
- Do not hardcode weld_param to a specific value. It must be configurable.
- Do not hardcode the weld strategy. It must be selectable.
- Do not add ray-cast to DP conditioning (still Phase 3 Q5).
- Do not add the additional obstacles (still Phase 3 Q6).
- Do not optimize for speed. Correctness first. The weld verification must pass
  before any integration work.

---

## Execution Order

```
Step 1: Read WRITEUP.md in full before writing any code.
  The welding math is precise and the implementation must match it exactly.
  Pay special attention to the knot vector structure and the Bézier extraction
  property at clamped endpoints.

Step 2: Implement pih_weld.py (Component 2)
  - weld_c2() with all three strategies
  - Pure math, no dependencies on env or solver
  - Write test_weld.py alongside it
  - Run Check 1: all 300 tests pass (100 splines × 3 strategies)

Step 3: Implement oracle KTO (Component 1)
  - PIHConfig.with_true_as_assumed()
  - oracle_plan_pih_with_kto()
  - Refactor plan_pih_with_kto() for start/remaining_waypoints if needed
  - Verify oracle produces feasible plans on both_wrong scenarios

Step 4: Implement trigger layer (Component 3)
  - PackageInHoleKTOController replan config
  - On-contact and periodic triggers
  - Integration with oracle + weld
  - Run Check 2: oracle replan recovers both_wrong episodes

Step 5: Implement data collection harness (Component 4)
  - collect_pih_phase2.py
  - Run Check 3: data integrity on 20 episodes

Step 6: Run Check 4 (Phase 1 regression)
  - replan_trigger="none" produces identical results to Phase 1 baseline

Step 7: Commit and tag as phase-2-complete
```

---

## Notes for Claude Code

1. **Read WRITEUP.md before writing any weld code.** The math is fully derived and
   the implementation should follow it exactly. Do not re-derive the formulas. Do
   not simplify or approximate. The verification tests compare against scipy's
   BSpline evaluation, so algebraic shortcuts that introduce numerical drift will
   be caught.

2. **pih_weld.py has zero dependencies on pih_env or pih_solver.** It is pure B-spline
   math. It takes numpy arrays in, returns numpy arrays out. This separation is
   deliberate — the weld function will eventually be called by DP inference code
   (Phase 3) and must not import environment-specific modules.

3. **The oracle replans from current state, not from episode start.** This is the
   most likely source of bugs. `plan_pih_with_kto()` probably assumes it starts
   from the initial pad position. The oracle needs to start from wherever the lander
   currently is, heading toward whatever waypoints remain. Read the existing function
   carefully and refactor if it hardcodes the start.

4. **Test the weld before integrating it.** Step 2 must be complete and passing before
   Step 4 begins. If the weld math is wrong, integration will produce subtle trajectory
   discontinuities that are hard to diagnose. Get the math right in isolation first.

5. **The data collection format matters for Phase 3.** The per-episode record structure
   is designed so Phase 3 training can reconstruct what happened at any point in the
   episode. Don't simplify the record structure even if some fields seem redundant now.
   Phase 3 experiments will use fields that Phase 2 doesn't need.

6. **On-contact trigger and periodic trigger can coexist.** If `replan_trigger="periodic"`
   and `replan_interval_s=2.0`, the periodic trigger fires every 2 seconds AND on
   contact. The replan_events list in the data record captures all of them.

7. **WRITEUP.md recommends point weld when you can apply immediately.** For the oracle
   KTO case (where the oracle computes a new plan and you want to switch to it), the
   point weld is the natural choice. Overlap strategies exist for when inference delay
   matters, which is a Phase 3 concern (DP inference latency). Phase 2 should
   implement all three but default to point weld.
