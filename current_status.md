# Simple Lander — Current Status

## Environment (`lunar_lander.py`)

**LunarLander** — a simplified gymnasium env with time-optimal reward (-dt per step).

- **Viewport**: 900x600 pixels, world is 30x20 units
- **Spawn**: uniform x over full display width `[1, 29]`, y gaussian near top ~17
- **Landing pad**: center of screen at (15, 5)
- **Obstacles**: 0-5 configurable satellite obstacles
- **Action space**: continuous `[-1, 1]` x 2 — `[main_engine, side_engine]`
  - Main: -1 = off, +1 = full thrust. Maps to `m_power = (a+1)/2`
  - Side: sign = direction, magnitude = power
- **Reward**: -dt per step. Crash/timeout -> total = -10s. Landing -> total ~ -elapsed_time
- **Observation**: 9-dim `[x, y, vx, vy, angle, angular_vel, leg1, leg2, sim_t]`

### Flight physics — leg spring damping

- Leg joint motors **always enabled** (matching Gymnasium LunarLander-v3)
- Legs kept at **full density** (1.0) throughout flight and landing
- Joint motors (LEG_SPRING_TORQUE=40, motorSpeed=±0.3) + joint constraints
  act as a linear angular damper on the lander body:
  **α_spring ≈ -4.75 · ω** (R² = 0.9998)
- This provides natural rotational stability, making keyboard control feel
  like v3 and improving PD tracking controller performance
- The damping is modelled as `LEG_SPRING_DAMPING = 4.75` in the surrogate
- **Pendulum coupling insight**: the motor/joint coupling also absorbs ~93%
  of side-thrust angular torque (effective G_side ≈ 0.037 vs bare 0.415,
  with sign reversal). However, the bare torque_arm/I model is kept for
  the solver/controller because:
  (a) the PD tracking controller compensates for the mismatch
  (b) the bare model produces well-scaled Fs values for translation
  (c) the pendulum model makes attitude nearly uncontrollable by Fs,
      requiring a fundamentally different control architecture

### Surrogate functions (in `lunar_lander.py`)

- `lander_dynamics(state, Fm, Fs)` — continuous-time ODE (6-state)
- `lander_acceleration(state, Fm, Fs)` — just the `[ax, ay, alpha]` accelerations
- `lander_step(state, Fm, Fs)` — one semi-implicit Euler step matching Box2D
- All use `LANDER_BODY_MASS = 4.817` (not system mass 4.959)
- Angular dynamics: `alpha = Fs * torque_arm / I - LEG_SPRING_DAMPING * omega`
- **Known model mismatch**: the Fs·torque_arm/I term overpredicts the angular
  effect of side thrust by ~15x (motors absorb most torque). The damping
  term is the dominant and accurate component.

## Solver (`solver.py`)

**KTO B-spline trajectory optimizer** using Drake's KinematicTrajectoryOptimization:

- Plans in `[x, y, theta]` space with cubic B-splines (15 control points)
- Dynamics constraints at ~65 sample points verify thrust feasibility via inverse dynamics
- Torque model includes leg spring damping and is **validated at import time**
- Two-phase solve: warm-start (no obstacles) -> obstacle phase
- Default time budget: 5s total, 1s warm-start

### Key constants

- `MASS = 4.817` (lander body), `INERTIA = 0.833`, `GRAVITY = 10.0`
- `THRUST_MAX ~ 86.7 N` (main), `SIDE_MAX ~ 30.0 N` (side)
- `SIDE_FORCE_MAX = 75.0 N` (action scaling for side engine)
- `LEG_DAMPING = 4.75` (angular damping from leg joints + motors)

### Tracking controller (`track()` + `_tracking_step()`)

Cascaded PD feedback: outer loop (position) -> inner loop (attitude).

- Gains: `Kp_pos=4, Kd_pos=4, Kp_att=50, Kd_att=10`
- Blends commanded theta (from acceleration vector) with plan theta: 40/60 split
- Inner loop accounts for damping: `Fs = (α_des + c·ω) · I / torque_arm`

## KTOController (in `lunar_lander.py`)

**Phase 1** — PD tracking of the KTO plan:
- Reads actual state from Box2D each step
- Calls `_tracking_step()` to compute corrective Fm/Fs
- Converts to actions: `a_main = 2*Fm/THRUST_MAX - 1`, `a_side = Fs/SIDE_FORCE_MAX`

**Phase 2** — Heuristic PD controller for final descent (after plan ends ~2m above pad).

## Running

```
python lunar_lander.py                        # heuristic controller
python lunar_lander.py --kto                  # KTO plan + PD tracking
python lunar_lander.py --kto --obstacles 3    # with obstacles
python lunar_lander.py --kto --seed 42        # specific seed
python lunar_lander.py --keyboard             # manual control
python lunar_lander.py --speedup 2.0          # 2x playback
```

## Test suite (`tests.py` — 24 tests)

| Test class | Count | What it checks |
|---|---|---|
| `TestSurrogateDynamics` | 4 | `lander_step` vs Box2D: freefall, main thrust, side thrust, three-way agreement with KTO plan |
| `TestPhysicsDivergence` | 4 | Euler step vs Box2D, solver plan replay, mass/inertia match |
| `TestKTOActions` | 3 | Action ranges, no dead zones |
| `TestKTOTracking` | 2 | Drift correlates with lateral demand, PD tracking vs direct impulse |
| `TestThrustTriangles` | 2 | Rendering indicators |
| `TestInitialState` | 2 | Spawn x uniform across display, y gaussian near top |
| `TestKTOLanding` | 1 | 10 episodes, >=6 land, faster than realtime |
| `TestLegSpringCharacterization` | 6 | Motor torque characterization, damping fits, total damping profile |
