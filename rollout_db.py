"""RolloutDB — SQLite storage for KTO rollout training data.

Stores per-episode obs/action trajectories, per-model-call conditioning
vectors and fitted target CPs (15 CPs × 2 = 30 dims), and CFG annotations.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    seed          INTEGER NOT NULL,
    outcome       TEXT NOT NULL,
    episode_length INTEGER NOT NULL,
    raw_length    INTEGER NOT NULL,
    sim_time      REAL NOT NULL,
    wall_time_ms  REAL NOT NULL,
    dt            REAL NOT NULL DEFAULT 0.02,
    action_horizon REAL NOT NULL DEFAULT 3.0,
    target_frequency REAL NOT NULL DEFAULT 5.0,
    n_denoising_steps INTEGER NOT NULL DEFAULT 10,
    num_obstacles INTEGER NOT NULL DEFAULT 0,
    guidance_margin REAL NOT NULL DEFAULT 0.2,

    obs_trajectory    BLOB NOT NULL,
    action_trajectory BLOB NOT NULL,

    terrain_x     TEXT,
    terrain_y     TEXT,
    obstacle_centers TEXT,

    call_steps    BLOB,
    call_cond     BLOB,
    call_output   BLOB,
    call_wp_t     BLOB,

    waypoint_json TEXT,

    cfg_wp_outcome BLOB,
    cfg_crash_t    BLOB,

    created_at    TEXT DEFAULT (datetime('now'))
);
"""

X_DIM = 30  # 15 CPs × 2
COND_DIM = 131


class RolloutDB:
    def __init__(self, db_path: str = "rollouts.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(_CREATE_TABLE)
        self.conn.commit()

    def save_episode(
        self,
        seed: int,
        outcome: str,
        episode_length: int,
        raw_length: int,
        sim_time: float,
        wall_time_ms: float,
        obs_trajectory: np.ndarray,
        action_trajectory: np.ndarray,
        call_steps: np.ndarray,
        call_cond: np.ndarray,
        call_output: np.ndarray,
        call_wp_t: np.ndarray,
        cfg_wp_outcome: np.ndarray,
        cfg_crash_t: np.ndarray,
        dt: float = 0.02,
        action_horizon: float = 3.0,
        target_frequency: float = 5.0,
        n_denoising_steps: int = 10,
        num_obstacles: int = 0,
        guidance_margin: float = 0.2,
        terrain_x: list | None = None,
        terrain_y: list | None = None,
        obstacle_centers: list | None = None,
        waypoint_json: dict | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO episodes (
                seed, outcome, episode_length, raw_length, sim_time, wall_time_ms,
                dt, action_horizon, target_frequency, n_denoising_steps,
                num_obstacles, guidance_margin,
                obs_trajectory, action_trajectory,
                terrain_x, terrain_y, obstacle_centers,
                call_steps, call_cond, call_output, call_wp_t,
                waypoint_json, cfg_wp_outcome, cfg_crash_t
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                seed, outcome, episode_length, raw_length, sim_time, wall_time_ms,
                dt, action_horizon, target_frequency, n_denoising_steps,
                num_obstacles, guidance_margin,
                obs_trajectory.astype(np.float32).tobytes(),
                action_trajectory.astype(np.float32).tobytes(),
                json.dumps(terrain_x) if terrain_x is not None else None,
                json.dumps(terrain_y) if terrain_y is not None else None,
                json.dumps(obstacle_centers) if obstacle_centers is not None else None,
                call_steps.astype(np.int32).tobytes(),
                call_cond.astype(np.float32).tobytes(),
                call_output.astype(np.float32).tobytes(),
                call_wp_t.astype(np.float32).tobytes(),
                json.dumps(waypoint_json) if waypoint_json is not None else None,
                cfg_wp_outcome.astype(np.int8).tobytes(),
                cfg_crash_t.astype(np.float32).tobytes(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def load_episode(self, episode_id: int) -> dict:
        cur = self.conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"Episode {episode_id} not found")

        cols = [desc[0] for desc in cur.description]
        d = dict(zip(cols, row))

        ep_len = d["episode_length"]
        n_calls = len(np.frombuffer(d["call_steps"], dtype=np.int32))

        d["obs_trajectory"] = np.frombuffer(
            d["obs_trajectory"], dtype=np.float32
        ).reshape(ep_len, 9).copy()
        d["action_trajectory"] = np.frombuffer(
            d["action_trajectory"], dtype=np.float32
        ).reshape(ep_len, 2).copy()
        d["call_steps"] = np.frombuffer(
            d["call_steps"], dtype=np.int32
        ).copy()
        d["call_cond"] = np.frombuffer(
            d["call_cond"], dtype=np.float32
        ).reshape(n_calls, COND_DIM).copy()
        d["call_output"] = np.frombuffer(
            d["call_output"], dtype=np.float32
        ).reshape(n_calls, X_DIM).copy()
        d["call_wp_t"] = np.frombuffer(
            d["call_wp_t"], dtype=np.float32
        ).copy()
        d["cfg_wp_outcome"] = np.frombuffer(
            d["cfg_wp_outcome"], dtype=np.int8
        ).copy()
        d["cfg_crash_t"] = np.frombuffer(
            d["cfg_crash_t"], dtype=np.float32
        ).copy()

        for key in ("terrain_x", "terrain_y", "obstacle_centers", "waypoint_json"):
            if d[key] is not None:
                d[key] = json.loads(d[key])

        return d

    def get_training_frames(self, episode_id: int):
        """Yield valid (cond, target_cps, cfg_wp, cfg_crash_t) per model call."""
        ep = self.load_episode(episode_id)
        action_horizon = ep["action_horizon"]
        dt = ep["dt"]
        raw_length = ep["raw_length"]
        outcome = ep["outcome"]
        t_end = raw_length * dt

        for i, step in enumerate(ep["call_steps"]):
            t_call = step * dt
            if outcome != "landed" and t_call + action_horizon > t_end + dt * 0.5:
                continue
            yield (
                ep["call_cond"][i],
                ep["call_output"][i],
                int(ep["cfg_wp_outcome"][i]),
                float(ep["cfg_crash_t"][i]),
            )

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM episodes")
        return cur.fetchone()[0]

    def count_by_outcome(self) -> dict:
        cur = self.conn.execute(
            "SELECT outcome, COUNT(*) FROM episodes GROUP BY outcome"
        )
        return dict(cur.fetchall())

    def summary(self) -> dict:
        cur = self.conn.execute(
            "SELECT outcome, COUNT(*), AVG(sim_time), AVG(episode_length) "
            "FROM episodes GROUP BY outcome"
        )
        results = {}
        for outcome, count, avg_time, avg_len in cur.fetchall():
            results[outcome] = {
                "count": count,
                "avg_sim_time": round(avg_time, 2),
                "avg_episode_length": round(avg_len, 1),
            }
        return results

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
