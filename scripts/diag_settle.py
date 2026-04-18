"""Diagnose the settling/ground-contact failure.

Measures:
- Leg fixture AABB lowest y at each step (actual contact geometry)
- Engine on/off state
- Controller ay_des
- both_gnd status

Run:
  docker run --rm pih-validate python scripts/diag_settle.py
"""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from kto_lander import LanderParams, TrackerGains, compound_inertia_about_body_com, DT
from kto_lander import MAIN_THRUST_MAX, M_POWER_MIN
from pih_env import PackageInHoleEnv, PIHConfig, DT as ENV_DT
from pih_solver import plan_pih_with_kto, PackageInHoleKTOController

SEED = 70000

cfg = PIHConfig(
    hole_depth=1.0,
    package_height_true=1.5,
    package_height_assumed=0.5,
    package_mass_true=2.0,
    package_mass_assumed=2.0,
)

env = PackageInHoleEnv(config=cfg, render_mode=None)
env.reset(seed=SEED)
uw = env.unwrapped
uw.lander.linearVelocity  = (0.0, 0.0)
uw.lander.angularVelocity = 0.0

x0 = float(uw.lander.position.x)
y0 = float(uw.lander.position.y)

mass, inertia = compound_inertia_about_body_com(env)
params = LanderParams(mass=mass, inertia=inertia)
print(f"[diag] base lander: mass={mass:.4f} kg  inertia={inertia:.4f}")
print(f"[diag] MAIN_THRUST_MAX={MAIN_THRUST_MAX:.2f} N  hover_base={mass*9.81:.2f} N  m_power_hover={mass*9.81/MAIN_THRUST_MAX:.3f}")
print(f"[diag] M_POWER_MIN={M_POWER_MIN}  MIN_THRUST={M_POWER_MIN*MAIN_THRUST_MAX:.2f} N")

plan, T, waypoints = plan_pih_with_kto(x0, y0, 0.0, 0.0, cfg, params, verbose=True)
ctrl = PackageInHoleKTOController(plan, waypoints, cfg, params)

print(f"\n[diag] plan T={T:.2f}s")
print(f"[diag] expected PD eq after T: y_eq = landing_y - kd_y*0.3/kp_y")

def leg_aabb_low(leg):
    """Lowest y of all fixtures on this leg body (world frame)."""
    lo = math.inf
    f = leg.fixtures
    for fix in (f if hasattr(f, '__iter__') else [f]):
        aabb = fix.shape.getAABB(leg.transform, 0)
        if aabb.lowerBound.y < lo:
            lo = aabb.lowerBound.y
    return lo

done = False
step_i = 0
attach_step = None

# Print header at settle phase start
print(f"\n{'step':>6} {'t':>6} {'ly':>7} {'leg0y':>7} {'leg1y':>7} {'bot0':>7} {'bot1':>7} {'gnd':>5} {'eng':>5} {'ay_des':>8} {'vy':>7} {'spd':>7}")

while not done:
    action, dbg = ctrl.step(env)
    _, _, term, trunc, info = env.step(action)
    done = term or trunc
    step_i += 1

    if uw._attached and attach_step is None:
        attach_step = step_i
        print(f"\n*** ATTACHED at step {step_i} ***")
        pm = ctrl.params.mass
        print(f"    ctrl.params.mass={pm:.4f}  hover_need={pm*9.81:.2f} N  m_power_req={pm*9.81/MAIN_THRUST_MAX:.3f}")

    # Print every step in the settling phase (last 400 steps of plan + 200 after)
    t = ctrl.t - DT  # time used for this step
    plan_end_step = int(T / DT)
    if step_i >= plan_end_step - 50:
        lander = uw.lander
        pos = lander.position
        vel = lander.linearVelocity
        ly = float(pos.y)
        vy = float(vel.y)
        spd = math.sqrt(float(vel.x)**2 + vy**2)

        l0, l1 = uw.legs[0], uw.legs[1]
        l0y = float(l0.worldCenter.y)
        l1y = float(l1.worldCenter.y)
        bot0 = leg_aabb_low(l0)
        bot1 = leg_aabb_low(l1)
        gnd = uw.legs[0].ground_contact or uw.legs[1].ground_contact
        eng_on = action[0] > 0.0

        # Recompute ay_des for diagnosis
        r = plan(min(t, T))
        xr, yr, vxr, vyr = r[0], r[1], r[2], r[3]
        ayr = r[7]
        ay_des = ayr + 8.0*(yr - ly) + 5.0*(vyr - vy)

        # Extra hover diagnostics for first 10 steps after plan ends
        extra = ""
        if step_i > plan_end_step and step_i <= plan_end_step + 10:
            l0 = uw.legs[0]
            extra = f"  awake={l0.awake} mot={l0.joint.motorEnabled} jang={math.degrees(l0.joint.angle):.2f}°"
        print(f"{step_i:>6} {t:>6.2f} {ly:>7.4f} {l0y:>7.4f} {l1y:>7.4f} {bot0:>7.4f} {bot1:>7.4f} {str(gnd):>5} {str(eng_on):>5} {ay_des:>8.3f} {vy:>7.4f} {spd:>7.4f}{extra}")

    if step_i > plan_end_step + 500:
        print("[diag] stopping after plan + 500 steps")
        break

reason = info.get("termination_reason", "n/a")
print(f"\n[diag] done step={step_i} reason={reason}")
env.close()
