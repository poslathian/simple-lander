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

### Flight physics tweaks

- Leg joint motors **disabled during flight** — no torque on lander from legs
- Leg density set to **0.001 during flight** — effectively massless, no drag on lander
- Both restored on ground contact for landing shock absorption
- This makes the lander behave as a clean single rigid body in flight

### Surrogate functions (in `lunar_lander.py`)

- `lander_dynamics(state, Fm, Fs)` — continuous-time ODE (6-state)
- `lander_acceleration(state, Fm, Fs)` — just the `[ax, ay, alpha]` accelerations
- `lander_step(state, Fm, Fs)` — one semi-implicit Euler step matching Box2D
- All use `LANDER_BODY_MASS = 4.817` (not system mass 4.959)

## Solver (`solver.py`)

**KTO B-spline trajectory optimizer** using Drake's KinematicTrajectoryOptimization:

- Plans in `[x, y, theta]` space with cubic B-splines (15 control points)
- Dynamics constraints at ~65 sample points verify thrust feasibility via inverse dynamics
- Torque model **validated at import time** against `lander_acceleration()`
- Two-phase solve: warm-start (no obstacles) -> obstacle phase
- Default time budget: 5s total, 1s warm-start

### Key constants

- `MASS = 4.817` (lander body), `INERTIA = 0.833`, `GRAVITY = 10.0`
- `THRUST_MAX ~ 86.7 N` (main), `SIDE_MAX ~ 30.0 N` (side)
- `SIDE_FORCE_MAX = 75.0 N` (action scaling for side engine)

### Tracking controller (`track()` + `_tracking_step()`)

Cascaded PD feedback: outer loop (position) -> inner loop (attitude).

- Gains: `Kp_pos=4, Kd_pos=4, Kp_att=50, Kd_att=10`
- Blends commanded theta (from acceleration vector) with plan theta: 40/60 split
- Achieves **0.09x** the tracking error of open-loop replay

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

## Test suite (`tests.py` — 19 tests)

| Test class | Count | What it checks |
|---|---|---|
| `TestSurrogateDynamics` | 4 | `lander_step` vs Box2D: freefall, main thrust, side thrust, three-way agreement with KTO plan |
| `TestPhysicsDivergence` | 4 | Euler step vs Box2D, solver plan replay, mass/inertia match |
| `TestKTOActions` | 3 | Action ranges, no dead zones |
| `TestKTOTracking` | 2 | Drift correlates with lateral demand, PD tracking vs direct impulse |
| `TestThrustTriangles` | 2 | Rendering indicators |
| `TestInitialState` | 2 | Spawn x uniform across display, y gaussian near top |
| `TestKTOLanding` | 1 | 10 episodes, >=3 land, faster than realtime |
