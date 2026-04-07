"""Modal-scaled DAgger pipeline: parallel rollouts + GPU training.

Usage:
    .venv/bin/python modal_dagger.py [--rounds 10] [--checkpoint position_model.pt]

Architecture:
    - Rollouts: parallelized across Modal CPU containers (8 concurrent)
    - Training: GPU (T4) with larger batches and more epochs
    - Orchestration: local script manages DBs and round logic
"""

import argparse
import os
import pickle
import sqlite3
import subprocess
import sys
import time
import uuid

import modal
import numpy as np
import torch

from model import DiffusionMLP, CosineSchedule, DDIMSampler, X_DIM, COND_DIM, STATE_DIM, CFG_START, CFG_END, N_CPS, N_CHANNELS

app = modal.App("position-dagger")

# ── Modal images ─────────────────────────────────────────────────────────

# Rollout image: needs Box2D, scipy, torch (CPU), the full env
rollout_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("swig", "build-essential")
    .pip_install(
        "torch", "numpy", "scipy", "gymnasium[box2d]", "swig",
        "drake>=1.51",
    )
)

# Training image: just torch + numpy + scipy (GPU)
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy")
)

# ── Source files to upload ───────────────────────────────────────────────

def _read_source_files():
    """Read all source files needed on Modal."""
    files = {}
    for name in [
        "model.py", "diffusion_controller.py", "lunar_lander.py",
        "solver.py", "thrust_spline.py", "eval.py", "collect.py",
        "guidance_controller.py",
    ]:
        if os.path.exists(name):
            with open(name, "rb") as f:
                files[name] = f.read()
    return files


def _write_source_files(files, workdir="/root/work"):
    """Write source files to workdir on Modal container."""
    os.makedirs(workdir, exist_ok=True)
    for name, content in files.items():
        with open(f"{workdir}/{name}", "wb") as f:
            f.write(content)
    return workdir


# ── Remote rollout function ──────────────────────────────────────────────

@app.function(
    image=rollout_image,
    timeout=600,
    cpu=2,
    memory=4096,
)
def rollout_batch(
    source_files: dict,
    checkpoint_bytes: bytes,
    seeds: list[int],
    margin_mean: float,
    outcome_cond: int = 1,
) -> list[dict]:
    """Run a batch of episodes on a single Modal container.

    Returns list of episode dicts with embedded frame data.
    """
    import sys
    workdir = _write_source_files(source_files)
    sys.path.insert(0, workdir)
    os.chdir(workdir)

    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.interpolate import make_lsq_spline

    # Reload modules in this container
    from model import DiffusionMLP, CosineSchedule, DDIMSampler
    from model import X_DIM, COND_DIM, STATE_DIM, CFG_START, CFG_END, N_CPS, N_CHANNELS
    from diffusion_controller import (
        KTODiffusionController, NoiseModel, Outcome,
        _build_cond, ObstacleRelative, WaypointTarget,
        DT, DEGREE,
    )
    import solver

    # Load model
    with open("/tmp/model.pt", "wb") as f:
        f.write(checkpoint_bytes)
    ckpt = torch.load("/tmp/model.pt", weights_only=False)
    model = DiffusionMLP()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32)
    schedule = CosineSchedule(T=ckpt.get("T", 100))
    sampler = DDIMSampler(model, schedule, n_steps=10)

    class _Model:
        def predict(self, cond, outcome, guidance_scale=2.0):
            full = np.zeros(COND_DIM, dtype=np.float32)
            full[:STATE_DIM] = cond[:STATE_DIM]
            full[CFG_START] = float(outcome)
            ct = torch.tensor(full, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                x_norm = sampler.sample_cfg(ct, guidance_scale=guidance_scale)
            x_raw = (x_norm * x_std + x_mean).squeeze(0).numpy()
            cps = x_raw.reshape(N_CPS, N_CHANNELS)
            cps[0] = [0, 0, 0]
            return cps

    live_model = _Model()

    # Fit KTO window helper
    def _fit_kto_window(plan, idx, action_horizon, dt=DT):
        n_steps = int(round(action_horizon / dt))
        end_idx = min(idx + n_steps, len(plan["x"]) - 1)
        if end_idx <= idx:
            return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        n = end_idx - idx
        t = np.linspace(0, (n - 1) * dt, n)
        x0 = float(plan["x"][idx])
        y0 = float(plan["y"][idx])
        th0 = float(plan["theta"][idx])
        x_rel = np.array(plan["x"][idx:end_idx], dtype=np.float64) - x0
        y_rel = np.array(plan["y"][idx:end_idx], dtype=np.float64) - y0
        th_rel = np.array(plan["theta"][idx:end_idx], dtype=np.float64) - th0
        if len(t) < N_CPS:
            return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        duration = t[-1]
        if duration < 1e-6:
            return np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        n_internal = N_CPS - DEGREE + 1
        internal = np.linspace(0, duration, n_internal)
        knots = np.concatenate([
            np.full(DEGREE + 1, 0.0), internal[1:-1], np.full(DEGREE + 1, duration),
        ])
        cps = np.zeros((N_CPS, N_CHANNELS), dtype=np.float64)
        for ch, vals in enumerate([x_rel, y_rel, th_rel]):
            try:
                spline = make_lsq_spline(t, vals, knots, k=DEGREE)
                cps[:, ch] = spline.c[:N_CPS]
            except Exception:
                pass
        return cps

    # Register env
    try:
        gym.register(id="LL-modal", entry_point="lunar_lander:LunarLander", max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-modal", render_mode=None, continuous=True)

    results = []
    outcome_enum = Outcome.SUCCESS if outcome_cond == 1 else Outcome.FAIL

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        ctrl = KTODiffusionController(
            env, model=live_model, target_frequency=3.0,
            action_horizon=1.5, outcome=outcome_enum,
        )
        ctrl.warm_start(time_budget=5.0)

        frames = []
        total_reward = 0.0
        done = False
        step_idx = 0
        last_inf = -999
        spi = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))

        while not done:
            t_sim = uw.elapsed_s
            L = uw.lander

            if step_idx - last_inf >= spi:
                q_now = (L.position.x, L.position.y, L.angle)
                vel = (L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity)
                q_prev = ctrl._last_inference_q if ctrl._last_inference_q else q_now
                dq = (solver.PAD_X - q_now[0], solver.PAD_Y - q_now[1], 0.0 - q_now[2])
                dq_prime = (0.0 - vel[0], -0.5 - vel[1], 0.0 - vel[2])
                kto_ref = ctrl._get_kto_ref(t_sim)
                guidance_q = kto_ref["q"] if kto_ref else q_now

                model_input = _build_cond(
                    t_obs_cmd_latency=DT, q_now=q_now, q_prev=q_prev,
                    obstacle=ObstacleRelative(0.0, 0.0, 0.0),
                    waypoint=WaypointTarget(dq=dq, dq_prime=dq_prime),
                    guidance_q=guidance_q, action_horizon=ctrl.action_horizon,
                )

                ctrl.inference()
                last_inf = step_idx

                model_out = np.zeros((N_CPS, N_CHANNELS), dtype=np.float32)
                if ctrl._diff_splines is not None:
                    for ch in range(N_CHANNELS):
                        model_out[:, ch] = ctrl._diff_splines[ch].c[:N_CPS]

                kto_idx = int(round((t_sim - ctrl._kto_t0) / DT))
                actual_cps = _fit_kto_window(ctrl._kto.plan, kto_idx, ctrl.action_horizon)

                frames.append({
                    "t_sim": t_sim,
                    "model_input": model_input.tobytes(),
                    "model_output_cps": model_out.flatten().tobytes(),
                    "actual_tracked_cps": actual_cps.astype(np.float32).flatten().tobytes(),
                })

            actual_margin = float(np.clip(abs(np.random.normal(margin_mean, 0.05)), 0.001, 1.0))
            tv, th = ctrl.get_action(guidance_margin=actual_margin)
            obs, reward, term, trunc, _ = env.step(np.array([tv, th], dtype=np.float32))
            total_reward += reward
            done = term or trunc
            step_idx += 1

        landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)

        results.append({
            "seed": seed,
            "outcome": 1 if landed else -1,
            "sim_duration": uw.elapsed_s,
            "reward": total_reward,
            "landed": landed,
            "frames": frames,
            "margin": margin_mean,
        })

    env.close()
    return results


# ── Remote training function ─────────────────────────────────────────────

@app.function(
    image=train_image,
    gpu="T4",
    timeout=1800,
)
def train_remote(
    model_py_bytes: bytes,
    checkpoint_bytes: bytes,
    frame_data: bytes,
    epochs: int,
    lr: float,
    batch_size: int,
    T: int,
    print_interval: int = 50,
) -> bytes:
    """Fine-tune model on GPU, return updated checkpoint bytes."""
    import sys
    import os
    import pickle

    os.makedirs("/root/train", exist_ok=True)
    with open("/root/train/model.py", "wb") as f:
        f.write(model_py_bytes)
    sys.path.insert(0, "/root/train")
    os.chdir("/root/train")

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from model import DiffusionMLP, CosineSchedule, X_DIM, COND_DIM, STATE_DIM, CFG_START, CFG_END

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load existing model
    with open("/tmp/ckpt.pt", "wb") as f:
        f.write(checkpoint_bytes)
    ckpt = torch.load("/tmp/ckpt.pt", weights_only=False, map_location=device)
    model = DiffusionMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load frame data
    frame_list = pickle.loads(frame_data)
    conds = np.array([f[0] for f in frame_list], dtype=np.float32)
    targets = np.array([f[1] for f in frame_list], dtype=np.float32)
    print(f"Training on {len(conds)} frames")

    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32).to(device)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32).to(device)

    # Update normalization stats with new data
    all_targets = targets
    new_mean = torch.tensor(all_targets.mean(axis=0), dtype=torch.float32).to(device)
    new_std = torch.tensor(all_targets.std(axis=0).clip(1e-6), dtype=torch.float32).to(device)
    # Blend old and new stats (weighted towards old for stability)
    alpha = 0.7
    x_mean = alpha * x_mean + (1 - alpha) * new_mean
    x_std = alpha * x_std + (1 - alpha) * new_std

    schedule = CosineSchedule(T=T)
    alpha_bar_all = torch.tensor(schedule.alpha_bar, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Build dataset inline
    class _DS(Dataset):
        def __init__(self, conds, targets):
            self.conds = conds
            self.targets = targets
        def __len__(self):
            return len(self.conds)
        def __getitem__(self, idx):
            cond = self.conds[idx].copy()
            if np.random.random() < 0.5:
                cond[CFG_START:CFG_END] = 0.0
            return (
                torch.tensor(cond, dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
            )

    loader = DataLoader(_DS(conds, targets), batch_size=batch_size,
                         shuffle=True, drop_last=False, num_workers=2, pin_memory=True)

    import time as _time
    t0 = _time.time()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for cond_b, target_b in loader:
            cond_b = cond_b.to(device, non_blocking=True)
            target_b = target_b.to(device, non_blocking=True)
            B = cond_b.shape[0]

            x0 = (target_b - x_mean) / x_std
            t = torch.randint(1, schedule.T + 1, (B,), device=device)
            ab = alpha_bar_all[t].unsqueeze(1)
            noise = torch.randn_like(x0)
            x_noisy = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
            eps_pred = model(x_noisy, cond_b, t)
            loss = nn.functional.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                x0_pred = x_noisy - eps_pred
                mse = nn.functional.mse_loss(x0_pred * x_std + x_mean, target_b).item()
                total_mse += mse
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches

        if (epoch + 1) % print_interval == 0 or epoch == 0:
            elapsed = _time.time() - t0
            print(f"  epoch {epoch+1:>4}/{epochs}  loss={avg_loss:.6f}  mse={avg_mse:.4f}  {elapsed:.0f}s")

    model.cpu()
    result = {
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean.cpu().numpy(),
        "x_std": x_std.cpu().numpy(),
        "epochs": ckpt.get("epochs", 0) + epochs,
        "n_frames": len(conds),
        "T": T,
    }
    import io
    buf = io.BytesIO()
    torch.save(result, buf)
    return buf.getvalue()


# ── Remote evaluation function ───────────────────────────────────────────

@app.function(
    image=rollout_image,
    timeout=600,
    cpu=2,
    memory=4096,
)
def eval_batch(
    source_files: dict,
    checkpoint_bytes: bytes,
    seeds: list[int],
    margin: float,
) -> dict:
    """Evaluate model at a single margin across seeds. Returns {margin, lands, total}."""
    import sys
    workdir = _write_source_files(source_files)
    sys.path.insert(0, workdir)
    os.chdir(workdir)

    import gymnasium as gym
    import numpy as np
    import torch
    from model import DiffusionMLP, CosineSchedule, DDIMSampler
    from model import COND_DIM, STATE_DIM, CFG_START, N_CPS, N_CHANNELS
    from diffusion_controller import KTODiffusionController, Outcome, DT

    # Load model
    with open("/tmp/model.pt", "wb") as f:
        f.write(checkpoint_bytes)
    ckpt = torch.load("/tmp/model.pt", weights_only=False)
    model = DiffusionMLP()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32)
    schedule = CosineSchedule(T=ckpt.get("T", 100))
    sampler = DDIMSampler(model, schedule, n_steps=10)

    class _Model:
        def predict(self, cond, outcome, guidance_scale=2.0):
            full = np.zeros(COND_DIM, dtype=np.float32)
            full[:STATE_DIM] = cond[:STATE_DIM]
            full[CFG_START] = float(outcome)
            ct = torch.tensor(full, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                x_norm = sampler.sample_cfg(ct, guidance_scale=guidance_scale)
            x_raw = (x_norm * x_std + x_mean).squeeze(0).numpy()
            cps = x_raw.reshape(N_CPS, N_CHANNELS)
            cps[0] = [0, 0, 0]
            return cps

    live = _Model()

    try:
        gym.register(id="LL-meval", entry_point="lunar_lander:LunarLander", max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-meval", render_mode=None, continuous=True)

    lands = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0
        ctrl = KTODiffusionController(env, model=live, target_frequency=3.0,
                                       action_horizon=1.5, outcome=Outcome.SUCCESS)
        ctrl.warm_start(time_budget=5.0)
        done, step_idx, last_inf = False, 0, -999
        spi = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))
        while not done:
            if step_idx - last_inf >= spi:
                ctrl.inference()
                last_inf = step_idx
            tv, th = ctrl.get_action(guidance_margin=margin)
            obs, r, term, trunc, _ = env.step(np.array([tv, th], dtype=np.float32))
            done = term or trunc
            step_idx += 1
        landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
        lands += landed
    env.close()

    return {"margin": margin, "lands": lands, "total": len(seeds)}


# ── Local DB (same schema as dagger_loop.py) ─────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    uid TEXT PRIMARY KEY, git_commit TEXT NOT NULL, env_seed INTEGER NOT NULL,
    diffusion_seed INTEGER, walltime_start REAL NOT NULL, outcome INTEGER NOT NULL,
    sim_duration REAL NOT NULL, guidance_margin REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    uid TEXT PRIMARY KEY, episode_uid TEXT NOT NULL, t_sim_frame REAL NOT NULL,
    model_input BLOB NOT NULL, model_output_cps BLOB NOT NULL,
    actual_tracked_cps BLOB NOT NULL,
    FOREIGN KEY (episode_uid) REFERENCES episodes(uid)
);
CREATE INDEX IF NOT EXISTS idx_frames_episode ON frames(episode_uid);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome);
"""


class DaggerDB:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)

    def clear(self):
        self.conn.execute("DELETE FROM frames")
        self.conn.execute("DELETE FROM episodes")
        self.conn.commit()

    def store_remote_results(self, results_list, git_commit):
        """Store results from remote rollout_batch calls."""
        for ep in results_list:
            ep_uid = str(uuid.uuid4())[:12]
            self.conn.execute(
                "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?)",
                (ep_uid, git_commit, ep["seed"], None, time.time(),
                 ep["outcome"], ep["sim_duration"], ep["margin"]),
            )
            for fr in ep["frames"]:
                self.conn.execute(
                    "INSERT INTO frames VALUES (?,?,?,?,?,?)",
                    (str(uuid.uuid4())[:12], ep_uid, fr["t_sim"],
                     fr["model_input"], fr["model_output_cps"],
                     fr["actual_tracked_cps"]),
                )
        self.conn.commit()

    def count_episodes(self, outcome=None):
        if outcome is None:
            return self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        return self.conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE outcome=?", (outcome,)
        ).fetchone()[0]

    def count_frames(self):
        return self.conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def sample_training_frames(self, n, landed_frac=0.8):
        """Sample n frames with outcome relabeling. Returns list of (cond_21dim, target_30dim)."""
        n_landed = int(n * landed_frac)
        n_failed = n - n_landed

        def _fetch(outcome, limit):
            rows = self.conn.execute(
                "SELECT f.model_input, f.actual_tracked_cps, e.outcome "
                "FROM frames f JOIN episodes e ON f.episode_uid = e.uid "
                "WHERE e.outcome = ? ORDER BY RANDOM() LIMIT ?",
                (outcome, limit),
            ).fetchall()
            result = []
            for inp_blob, cps_blob, actual_outcome in rows:
                inp = np.frombuffer(inp_blob, dtype=np.float32).copy()
                cps = np.frombuffer(cps_blob, dtype=np.float32).copy()
                # Build 21-dim cond with actual outcome (relabeling)
                cond = np.zeros(COND_DIM, dtype=np.float32)
                cond[:STATE_DIM] = inp[:STATE_DIM]
                cond[CFG_START] = float(actual_outcome)
                result.append((cond, cps))
            return result

        frames = _fetch(1, n_landed) + _fetch(-1, n_failed)
        return frames

    def copy_from(self, other):
        eps = other.conn.execute("SELECT * FROM episodes").fetchall()
        for row in eps:
            self.conn.execute("INSERT OR IGNORE INTO episodes VALUES (?,?,?,?,?,?,?,?)", row)
        frames = other.conn.execute("SELECT * FROM frames").fetchall()
        for row in frames:
            self.conn.execute("INSERT OR IGNORE INTO frames VALUES (?,?,?,?,?,?)", row)
        self.conn.commit()

    def close(self):
        self.conn.close()


# ── Orchestration ────────────────────────────────────────────────────────

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def _checkpoint_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def collect_parallel(source_files, ckpt_bytes, n_landed, n_failed,
                     margin_mean, seed_offset=10000):
    """Collect episodes in parallel on Modal until targets met.

    Launches multiple containers concurrently using rollout_batch.map().
    Each container runs a batch of seeds sequentially.
    """
    # Estimate total episodes needed (assume ~58% landing rate)
    est_total = int((n_landed + n_failed) / 0.55) + 30
    all_seeds = list(range(seed_offset, seed_offset + est_total))

    # Split into batches of 10 seeds each, run up to 8 containers concurrently
    batch_size = 10
    batches = [all_seeds[i:i+batch_size] for i in range(0, len(all_seeds), batch_size)]

    # Launch all batches concurrently via .map()
    map_args = [
        (source_files, ckpt_bytes, batch_seeds, margin_mean, 1)
        for batch_seeds in batches
    ]

    all_results = []
    landed = 0
    failed = 0

    for result_batch in rollout_batch.starmap(map_args):
        all_results.extend(result_batch)
        for ep in result_batch:
            if ep["landed"]:
                landed += 1
            else:
                failed += 1

        print(f"    Collected: {landed} landed, {failed} failed "
              f"(target: {n_landed}/{n_failed})")

        if landed >= n_landed and failed >= n_failed:
            break

    return all_results


def evaluate_parallel(source_files, ckpt_bytes, margins, seeds):
    """Evaluate model at multiple margins in parallel on Modal using fixed seed list."""

    # Launch all margin evaluations concurrently via starmap
    map_args = [
        (source_files, ckpt_bytes, seeds, margin)
        for margin in margins
    ]

    results = {}
    for r in eval_batch.starmap(map_args):
        results[r["margin"]] = (r["lands"], r["total"])
        rate = r["lands"] / r["total"]
        print(f"    margin={r['margin']:.3f}: {r['lands']}/{r['total']} = {rate:.0%}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--checkpoint", default="position_model.pt")
    parser.add_argument("--train-epochs", type=int, default=1000,
                        help="Training epochs per round (scaled up from local 500)")
    parser.add_argument("--train-frames", type=int, default=1000,
                        help="Frames sampled from archive per round (scaled up from 500)")
    parser.add_argument("--target-landed", type=int, default=80,
                        help="Landed episodes per collection (scaled up from 40)")
    parser.add_argument("--target-failed", type=int, default=20,
                        help="Failed episodes per collection (scaled up from 10)")
    parser.add_argument("--eval-seeds", type=int, default=50,
                        help="Seeds per margin for evaluation")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    git_commit = _git_commit()
    source_files = _read_source_files()
    T = 100

    print(f"{'='*60}")
    print(f"MODAL DAGGER PIPELINE")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Train epochs/round: {args.train_epochs}")
    print(f"  Frames/round: {args.train_frames}")
    print(f"  Collection target: {args.target_landed} landed + {args.target_failed} failed")
    print(f"  Eval seeds: {args.eval_seeds}")

    # Initialize DBs
    archive_db = DaggerDB("archive.db")
    current_db = DaggerDB("current.db")
    archive_db.clear()
    current_db.clear()

    # Fixed holdout seeds — same every round for apples-to-apples comparison
    HOLDOUT_SEEDS = list(range(90000, 90000 + args.eval_seeds))

    margin_mean = 0.2
    seed_counter = 10000
    ckpt_path = args.checkpoint

    with app.run():
        # ── Initial collection ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"INITIAL COLLECTION: margin_mean={margin_mean:.3f}")
        print(f"{'='*60}")

        ckpt_bytes = _checkpoint_bytes(ckpt_path)
        results = collect_parallel(
            source_files, ckpt_bytes,
            n_landed=args.target_landed, n_failed=args.target_failed,
            margin_mean=margin_mean, seed_offset=seed_counter,
        )
        seed_counter += len(results) + 50

        current_db.store_remote_results(results, git_commit)
        archive_db.copy_from(current_db)

        n_landed = sum(1 for r in results if r["landed"])
        n_total = len(results)
        rate = n_landed / n_total if n_total else 0
        print(f"  Initial: {n_landed}/{n_total} = {rate:.0%} (expected ~58%)")
        print(f"  Archive: {archive_db.count_frames()} frames")

        # ── DAgger rounds ────────────────────────────────────────────────
        margin_increment = 0.05

        for round_idx in range(args.rounds):
            t_round = time.time()
            print(f"\n{'─'*60}")
            print(f"ROUND {round_idx+1}/{args.rounds}  |  margin={margin_mean:.3f}  "
                  f"|  archive={archive_db.count_frames()} frames")
            print(f"{'─'*60}")

            # Step 1-2: Sample frames from archive with outcome relabeling
            print(f"\n  Sampling {args.train_frames} frames from archive...")
            frame_data = archive_db.sample_training_frames(
                args.train_frames, landed_frac=0.8
            )
            print(f"  Got {len(frame_data)} frames")

            # Step 3: Train on Modal GPU
            print(f"\n  Training {args.train_epochs} epochs on Modal GPU...")
            with open("model.py", "rb") as f:
                model_py_bytes = f.read()

            frame_data_bytes = pickle.dumps(frame_data)
            ckpt_bytes = _checkpoint_bytes(ckpt_path)

            new_ckpt_bytes = train_remote.remote(
                model_py_bytes=model_py_bytes,
                checkpoint_bytes=ckpt_bytes,
                frame_data=frame_data_bytes,
                epochs=args.train_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                T=T,
                print_interval=50,
            )

            # Save updated checkpoint
            ckpt_path = f"dagger_round{round_idx+1}.pt"
            with open(ckpt_path, "wb") as f:
                f.write(new_ckpt_bytes)
            print(f"  Saved: {ckpt_path}")

            # Step 4: Collect new episodes at candidate margin
            candidate_margin = np.clip(margin_mean + margin_increment, 0.001, 1.0)
            print(f"\n  Collecting at candidate margin={candidate_margin:.3f}...")
            current_db.clear()

            new_results = collect_parallel(
                source_files, new_ckpt_bytes,
                n_landed=args.target_landed, n_failed=args.target_failed,
                margin_mean=candidate_margin, seed_offset=seed_counter,
            )
            seed_counter += len(new_results) + 50

            current_db.store_remote_results(new_results, git_commit)
            n_landed = sum(1 for r in new_results if r["landed"])
            n_total = len(new_results)
            new_rate = n_landed / n_total if n_total else 0

            # Step 5: Evaluate
            print(f"\n  Evaluation (collection: {n_landed}/{n_total} = {new_rate:.0%}):")
            eval_margins = sorted(set([0.001, margin_mean, candidate_margin]))
            eval_results = evaluate_parallel(
                source_files, new_ckpt_bytes, eval_margins, HOLDOUT_SEEDS,
            )

            baseline_lands = eval_results.get(0.001, (0, len(HOLDOUT_SEEDS)))[0]
            candidate_key = min(eval_results.keys(),
                                key=lambda m: abs(m - candidate_margin))
            candidate_lands = eval_results[candidate_key][0]

            # Step 6: Add to archive
            archive_db.copy_from(current_db)
            print(f"\n  Archive: {archive_db.count_episodes()} episodes, "
                  f"{archive_db.count_frames()} frames")

            # Step 7-8: Decide margin advancement
            improved = new_rate >= 0.5 and candidate_lands >= baseline_lands

            if improved:
                margin_mean = candidate_margin
                print(f"  IMPROVED! margin_mean → {margin_mean:.3f}")
            else:
                print(f"  No improvement. margin_mean stays at {margin_mean:.3f}")

            elapsed = time.time() - t_round
            print(f"  Round took {elapsed:.0f}s")

        # ── Final evaluation sweep ───────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION")
        print(f"{'='*60}")
        final_ckpt = _checkpoint_bytes(ckpt_path)
        final_margins = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
        evaluate_parallel(
            source_files, final_ckpt, final_margins, HOLDOUT_SEEDS,
        )

    # Cleanup
    archive_db.close()
    current_db.close()

    print(f"\n{'='*60}")
    print(f"COMPLETE: final margin_mean={margin_mean:.3f}")
    print(f"Checkpoints: dagger_round1.pt .. dagger_round{args.rounds}.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
