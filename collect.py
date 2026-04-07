"""Collect training data: run KTODiffusionController with random margin < 0.4.

At each inference step, records:
  - 21-dim conditioning vector (20 state + 1 outcome)
  - 30-dim target CPs: KTO trajectory window fit to 10-CP position B-spline
  - Episode outcome: +1 landed, -1 crashed/timeout

Saves to a simple SQLite DB for training.
"""

import sqlite3
import sys
import time

import gymnasium as gym
import numpy as np
from scipy.interpolate import make_lsq_spline

from lunar_lander import LunarLander, KTOController, DT, TIMEOUT
from diffusion_controller import (
    KTODiffusionController, NoiseModel, Outcome,
    _build_cond, ObstacleRelative, WaypointTarget,
    N_CPS, N_CHANNELS, DEGREE, STATE_DIM, COND_DIM,
)
import solver


# ── DB setup ──────────────────────────────────────────────────────────────

_CREATE = """
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    cond BLOB NOT NULL,
    target_cps BLOB NOT NULL,
    outcome INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seed INTEGER NOT NULL,
    landed INTEGER NOT NULL,
    margin REAL NOT NULL,
    reward REAL NOT NULL,
    n_frames INTEGER NOT NULL
);
"""


class TrainingDB:
    def __init__(self, path="training.db"):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_CREATE)

    def save_episode(self, seed, landed, margin, reward, frames):
        """Save episode metadata + training frames."""
        cur = self.conn.execute(
            "INSERT INTO episodes (seed, landed, margin, reward, n_frames) VALUES (?,?,?,?,?)",
            (seed, int(landed), margin, reward, len(frames)),
        )
        ep_id = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO frames (episode_id, step, cond, target_cps, outcome) VALUES (?,?,?,?,?)",
            [(ep_id, f["step"], f["cond"].tobytes(), f["target_cps"].tobytes(), f["outcome"])
             for f in frames],
        )
        self.conn.commit()
        return ep_id

    def count_episodes(self):
        return self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def count_landed(self):
        return self.conn.execute("SELECT COUNT(*) FROM episodes WHERE landed=1").fetchone()[0]

    def count_frames(self):
        return self.conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def close(self):
        self.conn.close()


# ── Fit KTO window to position B-spline CPs ──────────────────────────────

def _fit_kto_window(plan, idx, action_horizon, dt=DT):
    """Fit KTO trajectory from idx forward (action_horizon seconds) to 10 CPs x 3.

    Returns (10, 3) numpy array of position control points in lander-relative coords.
    Origin = plan position at idx.
    """
    n_steps = int(round(action_horizon / dt))
    end_idx = min(idx + n_steps, len(plan["x"]) - 1)
    if end_idx <= idx:
        return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)

    # Time and position arrays for the window
    n = end_idx - idx
    t = np.linspace(0, (n - 1) * dt, n)
    x0, y0, th0 = float(plan["x"][idx]), float(plan["y"][idx]), float(plan["theta"][idx])
    x_rel = np.array(plan["x"][idx:end_idx], dtype=np.float64) - x0
    y_rel = np.array(plan["y"][idx:end_idx], dtype=np.float64) - y0
    th_rel = np.array(plan["theta"][idx:end_idx], dtype=np.float64) - th0

    if len(t) < N_CPS:
        # Not enough points, pad with zeros
        cps = np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        return cps

    # Build knot vector for LSQ fit
    duration = t[-1]
    if duration < 1e-6:
        return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)

    n_internal = N_CPS - DEGREE + 1
    internal = np.linspace(0, duration, n_internal)
    knots = np.concatenate([
        np.full(DEGREE + 1, 0.0),
        internal[1:-1],
        np.full(DEGREE + 1, duration),
    ])

    cps = np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
    for ch, vals in enumerate([x_rel, y_rel, th_rel]):
        try:
            spline = make_lsq_spline(t, vals, knots, k=DEGREE)
            cps[:, ch] = spline.c[:N_CPS]
        except Exception:
            pass  # Leave as zeros

    return cps


# ── Run one episode ───────────────────────────────────────────────────────

def run_episode(env, seed, margin):
    """Run episode, collect training frames at each inference step."""
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    uw.lander.linearVelocity = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0

    ctrl = KTODiffusionController(
        env, model=NoiseModel(), target_frequency=3.0,
        action_horizon=1.5, outcome=Outcome.SUCCESS,
    )
    ctrl.warm_start(time_budget=5.0)

    frames = []
    total_reward = 0.0
    done = False
    step_idx = 0
    last_inference_step = -999
    steps_per_inference = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))

    while not done:
        t_sim = uw.elapsed_s
        L = uw.lander

        # Inference step — collect training frame
        if step_idx - last_inference_step >= steps_per_inference:
            ctrl.inference()
            last_inference_step = step_idx

            # Build conditioning
            q_now = (L.position.x, L.position.y, L.angle)
            vel = (L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity)
            q_prev = ctrl._last_inference_q if ctrl._last_inference_q else q_now
            pad_x, pad_y = solver.PAD_X, solver.PAD_Y
            dq = (pad_x - q_now[0], pad_y - q_now[1], 0.0 - q_now[2])
            dq_prime = (0.0 - vel[0], -0.5 - vel[1], 0.0 - vel[2])

            kto_ref = ctrl._get_kto_ref(t_sim)
            guidance_q = kto_ref["q"] if kto_ref else q_now

            cond = _build_cond(
                t_obs_cmd_latency=DT,
                q_now=q_now, q_prev=q_prev,
                obstacle=ObstacleRelative(0.0, 0.0, 0.0),
                waypoint=WaypointTarget(dq=dq, dq_prime=dq_prime),
                guidance_q=guidance_q,
                action_horizon=ctrl.action_horizon,
            )

            # Target CPs: fit KTO trajectory window
            kto_idx = int(round((t_sim - ctrl._kto_t0) / DT))
            target_cps = _fit_kto_window(
                ctrl._kto.plan, kto_idx, ctrl.action_horizon,
            )

            frames.append({
                "step": step_idx,
                "cond": cond.astype(np.float32),
                "target_cps": target_cps.flatten().astype(np.float32),
                "outcome": 0,  # placeholder, set after episode
            })

        tv, th = ctrl.get_action(guidance_margin=margin)
        action_out = np.array([tv, th], dtype=np.float32)
        obs, reward, term, trunc, _ = env.step(action_out)
        total_reward += reward
        done = term or trunc
        step_idx += 1

    landed = (
        not uw.game_over
        and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
    )

    # Set outcome for all frames: +1 landed, -1 crashed/timeout
    outcome = 1 if landed else -1
    for f in frames:
        f["outcome"] = outcome

    return total_reward, landed, frames


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="training.db")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-margin", type=float, default=0.3)
    parser.add_argument("--seed-offset", type=int, default=5000)
    parser.add_argument("--min-success-rate", type=float, default=0.5)
    args = parser.parse_args()

    gym.register(
        id="LL-collect",
        entry_point="lunar_lander:LunarLander",
        max_episode_steps=1000,
    )
    env = gym.make("LL-collect", render_mode=None, continuous=True)
    db = TrainingDB(args.db)

    print(f"Collecting {args.episodes} episodes, margin in [0, {args.max_margin}]")
    print(f"Saving to {args.db}\n")

    t0 = time.time()
    landed_count = 0
    total_frames = 0

    for i in range(args.episodes):
        seed = args.seed_offset + i
        margin = np.random.uniform(0, args.max_margin)
        reward, landed, frames = run_episode(env, seed, margin)
        landed_count += landed
        total_frames += len(frames)

        db.save_episode(seed, landed, margin, reward, frames)

        status = "LANDED" if landed else "FAILED"
        if (i + 1) % 10 == 0:
            rate = landed_count / (i + 1)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{args.episodes}]  {status}  margin={margin:.3f}  "
                  f"land_rate={rate:.0%}  frames={total_frames}  "
                  f"{(i+1)/elapsed:.1f} ep/s")

    env.close()

    land_rate = landed_count / args.episodes
    print(f"\n{'='*50}")
    print(f"Collection complete")
    print(f"{'='*50}")
    print(f"Episodes: {args.episodes}")
    print(f"Landing rate: {land_rate:.0%} ({landed_count}/{args.episodes})")
    print(f"Total frames: {total_frames}")
    print(f"DB: {args.db}")
    print(f"{'='*50}")

    if land_rate < args.min_success_rate:
        print(f"WARNING: Landing rate {land_rate:.0%} below {args.min_success_rate:.0%}")
        return 1

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
