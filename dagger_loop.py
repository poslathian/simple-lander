"""DAgger training loop for position diffusion controller.

Schema: ArchiveDB and CurrentDB with identical episode + frame tables.
Procedure: collect → train on archive → evaluate → advance margin if improved.
"""

import hashlib
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import make_lsq_spline
from torch.utils.data import Dataset, DataLoader

from diffusion_controller import (
    KTODiffusionController, NoiseModel, Outcome,
    _build_cond, ObstacleRelative, WaypointTarget,
    N_CPS, N_CHANNELS, DEGREE, STATE_DIM, COND_DIM, DT,
)
from eval import TrainedModel, run_episode
from lunar_lander import LunarLander, KTOController, TIMEOUT
from model import (
    DiffusionMLP, CosineSchedule, DDIMSampler,
    X_DIM, CFG_START, CFG_END,
)
import solver


# ── DB Schema ────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    uid TEXT PRIMARY KEY,
    git_commit TEXT NOT NULL,
    env_seed INTEGER NOT NULL,
    diffusion_seed INTEGER,
    walltime_start REAL NOT NULL,
    outcome INTEGER NOT NULL,
    sim_duration REAL NOT NULL,
    guidance_margin REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    uid TEXT PRIMARY KEY,
    episode_uid TEXT NOT NULL,
    t_sim_frame REAL NOT NULL,
    model_input BLOB NOT NULL,
    model_output_cps BLOB NOT NULL,
    actual_tracked_cps BLOB NOT NULL,
    FOREIGN KEY (episode_uid) REFERENCES episodes(uid)
);

CREATE INDEX IF NOT EXISTS idx_frames_episode ON frames(episode_uid);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome);
"""


class DaggerDB:
    """Identically-shaped DB for both Archive and Current."""

    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)

    def clear(self):
        self.conn.execute("DELETE FROM frames")
        self.conn.execute("DELETE FROM episodes")
        self.conn.commit()

    def save_episode(self, uid, git_commit, env_seed, diffusion_seed,
                     walltime_start, outcome, sim_duration, guidance_margin):
        self.conn.execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?)",
            (uid, git_commit, env_seed, diffusion_seed,
             walltime_start, outcome, sim_duration, guidance_margin),
        )
        self.conn.commit()

    def save_frame(self, uid, episode_uid, t_sim_frame,
                   model_input, model_output_cps, actual_tracked_cps):
        self.conn.execute(
            "INSERT INTO frames VALUES (?,?,?,?,?,?)",
            (uid, episode_uid, t_sim_frame,
             model_input.astype(np.float32).tobytes(),
             model_output_cps.astype(np.float32).tobytes(),
             actual_tracked_cps.astype(np.float32).tobytes()),
        )

    def commit(self):
        self.conn.commit()

    def copy_from(self, other: "DaggerDB"):
        """Copy all episodes and frames from another DB."""
        eps = other.conn.execute("SELECT * FROM episodes").fetchall()
        for row in eps:
            self.conn.execute(
                "INSERT OR IGNORE INTO episodes VALUES (?,?,?,?,?,?,?,?)", row
            )
        frames = other.conn.execute("SELECT * FROM frames").fetchall()
        for row in frames:
            self.conn.execute(
                "INSERT OR IGNORE INTO frames VALUES (?,?,?,?,?,?)", row
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

    def sample_frames(self, n, landed_frac=0.8):
        """Sample n frames: landed_frac from landed episodes, rest from failed."""
        n_landed = int(n * landed_frac)
        n_failed = n - n_landed

        landed = self.conn.execute(
            "SELECT f.uid, f.episode_uid, f.t_sim_frame, f.model_input, "
            "f.model_output_cps, f.actual_tracked_cps "
            "FROM frames f JOIN episodes e ON f.episode_uid = e.uid "
            "WHERE e.outcome = 1 ORDER BY RANDOM() LIMIT ?",
            (n_landed,),
        ).fetchall()

        failed = self.conn.execute(
            "SELECT f.uid, f.episode_uid, f.t_sim_frame, f.model_input, "
            "f.model_output_cps, f.actual_tracked_cps "
            "FROM frames f JOIN episodes e ON f.episode_uid = e.uid "
            "WHERE e.outcome = -1 ORDER BY RANDOM() LIMIT ?",
            (n_failed,),
        ).fetchall()

        return landed + failed

    def get_episode_outcome(self, episode_uid):
        row = self.conn.execute(
            "SELECT outcome FROM episodes WHERE uid=?", (episode_uid,)
        ).fetchone()
        return row[0] if row else None

    def close(self):
        self.conn.close()


# ── Helpers ──────────────────────────────────────────────────────────────

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def _fit_kto_window(plan, idx, action_horizon, dt=DT):
    """Fit KTO trajectory window to 10 CP x 3 position B-spline."""
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
            pass
    return cps


# ── Collect episodes with frame recording ────────────────────────────────

def collect_episode(env, seed, model, margin, outcome_cond=Outcome.SUCCESS):
    """Run one episode, return (episode_info, frames_list).

    Each frame captures:
      - model_input: the 20-dim conditioning vector
      - model_output_cps: the (10,3) CPs the diffusion model produced
      - actual_tracked_cps: the (10,3) KTO window CPs (ground truth for PD tracking)
    """
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    uw.lander.linearVelocity = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0

    ctrl = KTODiffusionController(
        env, model=model, target_frequency=3.0,
        action_horizon=1.5, outcome=outcome_cond,
    )
    ctrl.warm_start(time_budget=5.0)

    frames = []
    total_reward = 0.0
    done = False
    step_idx = 0
    last_inference_step = -999
    steps_per_inference = max(1, int(round(1.0 / (ctrl.target_frequency * DT))))

    # Track the last diffusion output for frame recording
    last_model_output = np.zeros((N_CPS, N_CHANNELS), dtype=np.float32)

    while not done:
        t_sim = uw.elapsed_s
        L = uw.lander

        if step_idx - last_inference_step >= steps_per_inference:
            # Build conditioning BEFORE inference (to capture model input)
            q_now = (L.position.x, L.position.y, L.angle)
            vel = (L.linearVelocity.x, L.linearVelocity.y, L.angularVelocity)
            q_prev = ctrl._last_inference_q if ctrl._last_inference_q else q_now
            dq = (solver.PAD_X - q_now[0], solver.PAD_Y - q_now[1], 0.0 - q_now[2])
            dq_prime = (0.0 - vel[0], -0.5 - vel[1], 0.0 - vel[2])
            kto_ref = ctrl._get_kto_ref(t_sim)
            guidance_q = kto_ref["q"] if kto_ref else q_now

            model_input = _build_cond(
                t_obs_cmd_latency=DT,
                q_now=q_now, q_prev=q_prev,
                obstacle=ObstacleRelative(0.0, 0.0, 0.0),
                waypoint=WaypointTarget(dq=dq, dq_prime=dq_prime),
                guidance_q=guidance_q,
                action_horizon=ctrl.action_horizon,
            )

            # Run inference (updates ctrl._diff_splines)
            ctrl.inference()
            last_inference_step = step_idx

            # Capture model output: reconstruct CPs from the splines
            if ctrl._diff_splines is not None:
                # The model output is what was passed to _make_position_spline
                # We can recover it from the spline coefficients
                model_out = np.zeros((N_CPS, N_CHANNELS), dtype=np.float32)
                for ch in range(N_CHANNELS):
                    model_out[:, ch] = ctrl._diff_splines[ch].c[:N_CPS]
                last_model_output = model_out

            # Clamp diffusion CPs to within normalized margin of KTO CPs.
            # margin ∈ [0,1]: 0=pure KTO, 1=unclamped diffusion.
            # Radius = m/(1-m), matching get_action() semantics.
            kto_idx = int(round((t_sim - ctrl._kto_t0) / DT))
            kto_cps = _fit_kto_window(ctrl._kto.plan, kto_idx, ctrl.action_horizon)
            m = float(np.clip(margin, 0.0, 1.0))
            if m >= 1.0:
                actual_cps = last_model_output.copy()
            else:
                r = m / (1.0 - m)
                actual_cps = np.clip(last_model_output, kto_cps - r, kto_cps + r)

            frames.append({
                "t_sim": t_sim,
                "model_input": model_input.copy(),
                "model_output_cps": last_model_output.copy(),
                "actual_tracked_cps": actual_cps.astype(np.float32),
            })

        # Apply margin with Gaussian sampling
        actual_margin = np.clip(np.abs(np.random.normal(margin, 0.05)), 0.001, 1.0)
        tv, th = ctrl.get_action(guidance_margin=actual_margin)
        action_out = np.array([tv, th], dtype=np.float32)
        obs, reward, term, trunc, _ = env.step(action_out)
        total_reward += reward
        done = term or trunc
        step_idx += 1

    landed = (
        not uw.game_over
        and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
    )

    ep_info = {
        "seed": seed,
        "outcome": 1 if landed else -1,
        "sim_duration": uw.elapsed_s,
        "reward": total_reward,
        "landed": landed,
    }
    return ep_info, frames


def collect_until(env, model, target_landed, target_failed, margin_mean,
                  seed_offset=10000, outcome_cond=Outcome.SUCCESS):
    """Collect episodes until we have enough landed and failed."""
    episodes = []
    all_frames = []
    landed_count = 0
    failed_count = 0
    seed = seed_offset

    while landed_count < target_landed or failed_count < target_failed:
        ep_info, frames = collect_episode(
            env, seed, model, margin_mean, outcome_cond=outcome_cond,
        )
        episodes.append(ep_info)
        all_frames.append(frames)

        if ep_info["landed"]:
            landed_count += 1
        else:
            failed_count += 1

        seed += 1

        if seed - seed_offset > 500:
            print(f"  WARNING: 500 episodes and only {landed_count} landed, "
                  f"{failed_count} failed. Breaking.")
            break

    return episodes, all_frames


def store_episodes(db, episodes, all_frames, git_commit, margin_mean):
    """Store collected episodes and frames into a DaggerDB."""
    for ep_info, frames in zip(episodes, all_frames):
        ep_uid = str(uuid.uuid4())[:12]
        db.save_episode(
            uid=ep_uid,
            git_commit=git_commit,
            env_seed=ep_info["seed"],
            diffusion_seed=None,
            walltime_start=time.time(),
            outcome=ep_info["outcome"],
            sim_duration=ep_info["sim_duration"],
            guidance_margin=margin_mean,
        )
        for fr in frames:
            db.save_frame(
                uid=str(uuid.uuid4())[:12],
                episode_uid=ep_uid,
                t_sim_frame=fr["t_sim"],
                model_input=fr["model_input"],
                model_output_cps=fr["model_output_cps"].flatten(),
                actual_tracked_cps=fr["actual_tracked_cps"].flatten(),
            )
    db.commit()


# ── Training on archive frames ───────────────────────────────────────────

class ArchiveDataset(Dataset):
    """Dataset from sampled archive frames with outcome relabeling."""

    def __init__(self, frame_rows, archive_db: DaggerDB):
        self.conds = []
        self.targets = []

        for row in frame_rows:
            # row: (uid, episode_uid, t_sim_frame, model_input_blob,
            #        model_output_cps_blob, actual_tracked_cps_blob)
            episode_uid = row[1]
            model_input = np.frombuffer(row[3], dtype=np.float32).copy()
            actual_cps = np.frombuffer(row[5], dtype=np.float32).copy()

            # Step 2: Transform outcome conditioning to actual episode outcome
            actual_outcome = archive_db.get_episode_outcome(episode_uid)

            # Build full 21-dim conditioning with actual outcome
            full_cond = np.zeros(COND_DIM, dtype=np.float32)
            full_cond[:STATE_DIM] = model_input[:STATE_DIM]
            full_cond[CFG_START] = float(actual_outcome)  # relabel!

            self.conds.append(full_cond)
            self.targets.append(actual_cps)

        self.conds = np.array(self.conds, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.conds)

    def __getitem__(self, idx):
        cond = self.conds[idx].copy()
        # CFG dropout: zero outcome with 50% probability
        if np.random.random() < 0.5:
            cond[CFG_START:CFG_END] = 0.0
        return (
            torch.tensor(cond, dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def train_on_archive(model, archive_db, epochs=500, batch_size=64, lr=1e-4,
                     x_mean=None, x_std=None, T=100, print_interval=50,
                     n_frames=500):
    """Train model for additional epochs on sampled archive frames."""
    frame_rows = archive_db.sample_frames(n_frames, landed_frac=0.8)
    if len(frame_rows) < 10:
        print("  WARNING: Too few frames to train on")
        return x_mean, x_std

    dataset = ArchiveDataset(frame_rows, archive_db)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    schedule = CosineSchedule(T=T)

    # Compute normalization stats from this batch if not provided
    if x_mean is None:
        x_mean = torch.tensor(dataset.targets.mean(axis=0), dtype=torch.float32)
        x_std = torch.tensor(dataset.targets.std(axis=0).clip(1e-6), dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"  Training on {len(dataset)} frames for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for cond, target in loader:
            B = cond.shape[0]
            x0 = (target - x_mean) / x_std
            t = torch.randint(1, schedule.T + 1, (B,))
            alpha_bar = torch.tensor(
                [schedule.get_alpha_bar(ti.item()) for ti in t]
            ).unsqueeze(1)
            noise = torch.randn_like(x0)
            x_noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
            eps_pred = model(x_noisy, cond, t)
            loss = nn.functional.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Raw MSE on target prediction (for monitoring)
            with torch.no_grad():
                # Approximate x0 prediction at t=1
                x0_pred = x_noisy - eps_pred  # rough
                mse = nn.functional.mse_loss(x0_pred * x_std + x_mean, target).item()
                total_mse += mse
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches

        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"    epoch {epoch+1:>4}/{epochs}  loss={avg_loss:.6f}  mse={avg_mse:.4f}")

    model.eval()
    return x_mean, x_std


# ── Evaluation ───────────────────────────────────────────────────────────

class _LiveModel:
    """Wraps a live DiffusionMLP for evaluation (no checkpoint needed)."""

    def __init__(self, model, x_mean, x_std, T=100):
        self.model = model
        self.model.eval()
        schedule = CosineSchedule(T=T)
        self.sampler = DDIMSampler(model, schedule, n_steps=10)
        self.x_mean = x_mean
        self.x_std = x_std

    def predict(self, cond, outcome, guidance_scale=2.0):
        full = np.zeros(COND_DIM, dtype=np.float32)
        full[:STATE_DIM] = cond[:STATE_DIM]
        full[CFG_START] = float(outcome)
        ct = torch.tensor(full, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            x_norm = self.sampler.sample_cfg(ct, guidance_scale=guidance_scale)
        x_raw = (x_norm * self.x_std + self.x_mean).squeeze(0).numpy()
        cps = x_raw.reshape(N_CPS, N_CHANNELS)
        cps[0] = [0, 0, 0]
        return cps


def evaluate_model(env, model_wrapper, margins, n_seeds=30, seed_offset=20000):
    """Evaluate model at given margins. Returns {margin: (lands, total)}."""
    seeds = list(range(seed_offset, seed_offset + n_seeds))
    return evaluate_model_seeds(env, model_wrapper, margins, seeds)


def evaluate_model_seeds(env, model_wrapper, margins, seeds):
    """Evaluate model at given margins on explicit seed list. Returns {margin: (lands, total)}."""
    results = {}
    n_seeds = len(seeds)

    for margin in margins:
        lands = 0
        for seed in seeds:
            _, landed = run_episode(env, seed, model_wrapper, margin)
            lands += landed
        rate = lands / n_seeds
        results[margin] = (lands, n_seeds)
        print(f"    margin={margin:.3f}: {lands}/{n_seeds} = {rate:.0%}")

    return results


# ── Main DAgger Loop ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="position_model.pt")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--resume-margin", type=float, default=None,
                        help="Resume from this margin (skip initial collection, keep archive)")
    parser.add_argument("--resume-round", type=int, default=0,
                        help="Round number offset for checkpoint naming")
    parser.add_argument("--seed-offset", type=int, default=10000,
                        help="Starting seed for collection")
    # Training hyperparameters
    parser.add_argument("--train-frames", type=int, default=500,
                        help="Frames sampled from archive per round")
    parser.add_argument("--train-epochs", type=int, default=500,
                        help="Training epochs per round")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Model size
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden dim for DiffusionMLP")
    parser.add_argument("--n-blocks", type=int, default=6,
                        help="Number of residual blocks")
    # Output isolation
    parser.add_argument("--run-dir", default=".",
                        help="Directory for DBs and checkpoints")
    args = parser.parse_args()

    git_commit = _git_commit()
    T = 100

    # Create run directory for isolated output
    os.makedirs(args.run_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"DAgger config: frames={args.train_frames} epochs={args.train_epochs} "
          f"bs={args.batch_size} lr={args.lr}")
    print(f"  model: hidden={args.hidden} blocks={args.n_blocks}")
    print(f"  run_dir={args.run_dir}")
    print(f"{'='*60}")

    # Register env
    env_id = f"LL-dagger-{os.getpid()}"
    try:
        gym.register(
            id=env_id,
            entry_point="lunar_lander:LunarLander",
            max_episode_steps=1000,
        )
    except Exception:
        pass
    env = gym.make(env_id, render_mode=None, continuous=True)

    # Load existing model checkpoint
    checkpoint_path = args.checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = DiffusionMLP(hidden=args.hidden, n_blocks=args.n_blocks)
    # Load state dict — handle size mismatch for fresh larger models
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        print(f"  WARNING: state dict mismatch ({e}), initializing fresh model")
        # Keep x_mean/x_std from checkpoint but train from scratch
        pass
    model.eval()
    x_mean = torch.tensor(checkpoint["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(checkpoint["x_std"], dtype=torch.float32)
    print(f"  Loaded: {checkpoint.get('epochs', '?')} epochs, "
          f"{checkpoint.get('n_frames', '?')} frames, "
          f"margin={checkpoint.get('margin_mean', '?')}")

    # Wrap for collection/evaluation
    live_model = _LiveModel(model, x_mean, x_std, T=T)

    # Initialize DBs
    archive_db = DaggerDB(os.path.join(args.run_dir, "archive.db"))
    current_db = DaggerDB(os.path.join(args.run_dir, "current.db"))

    resuming = args.resume_margin is not None
    if not resuming:
        archive_db.clear()
        current_db.clear()
        print("Initialized ArchiveDB and CurrentDB (empty)")
    else:
        current_db.clear()
        print(f"RESUMING: keeping archive ({archive_db.count_episodes()} episodes, "
              f"{archive_db.count_frames()} frames), clearing current")

    # ── Initial collection: 40 landed + 10 failed ────────────────────────
    margin_mean = args.resume_margin if resuming else 0.2
    seed_counter = args.seed_offset

    if not resuming:
        print(f"\n{'='*60}")
        print(f"INITIAL COLLECTION: margin_mean={margin_mean:.3f}")
        print(f"{'='*60}")

        episodes, all_frames = collect_until(
            env, live_model,
            target_landed=40, target_failed=10,
            margin_mean=margin_mean, seed_offset=seed_counter,
        )
        seed_counter += len(episodes) + 10

        n_landed = sum(1 for e in episodes if e["landed"])
        n_total = len(episodes)
        landing_rate = n_landed / n_total if n_total > 0 else 0
        print(f"  Collected {n_total} episodes: {n_landed} landed, {n_total - n_landed} failed")
        print(f"  Landing rate: {landing_rate:.0%}")

        store_episodes(current_db, episodes, all_frames, git_commit, margin_mean)
        archive_db.copy_from(current_db)
        print(f"  Archive: {archive_db.count_episodes()} episodes, "
              f"{archive_db.count_frames()} frames")
    else:
        print(f"\n  Skipping initial collection (resuming at margin={margin_mean:.3f})")

    # ── DAgger iterations ────────────────────────────────────────────────
    n_rounds = args.rounds
    margin_increment = 0.05
    kto_baseline_margin = 0.001
    # Fixed holdout seeds — same every round for apples-to-apples comparison
    HOLDOUT_SEEDS = list(range(90000, 90050))

    print(f"\n{'='*60}")
    print(f"Starting {n_rounds} DAgger iterations")
    print(f"{'='*60}")

    round_offset = args.resume_round
    for round_idx in range(n_rounds):
        round_num = round_offset + round_idx + 1
        print(f"\n{'─'*60}")
        print(f"ROUND {round_num} (iter {round_idx+1}/{n_rounds})  |  margin_mean={margin_mean:.3f}  "
              f"|  archive={archive_db.count_frames()} frames")
        print(f"{'─'*60}")
        t_round = time.time()

        # Step 1-2: Sample frames from archive, relabel outcomes, train
        print(f"\n  Step 1-3: Training on {args.train_frames} archive frames, "
              f"{args.train_epochs} epochs...")
        model.train()
        x_mean, x_std = train_on_archive(
            model, archive_db,
            epochs=args.train_epochs, batch_size=args.batch_size, lr=args.lr,
            x_mean=x_mean, x_std=x_std, T=T,
            print_interval=max(50, args.train_epochs // 10),
            n_frames=args.train_frames,
        )
        model.eval()
        live_model = _LiveModel(model, x_mean, x_std, T=T)

        # Step 4: Collect new set with incremented margin
        candidate_margin = np.clip(margin_mean + margin_increment, 0.001, 1.0)
        print(f"\n  Step 4: Collecting at candidate margin={candidate_margin:.3f}...")
        current_db.clear()
        episodes, all_frames = collect_until(
            env, live_model,
            target_landed=40, target_failed=10,
            margin_mean=candidate_margin,
            seed_offset=seed_counter,
            outcome_cond=Outcome.SUCCESS,
        )
        seed_counter += len(episodes) + 10

        n_landed = sum(1 for e in episodes if e["landed"])
        n_total = len(episodes)
        new_rate = n_landed / n_total if n_total > 0 else 0
        n_frames_new = sum(len(f) for f in all_frames)

        # Store in current
        store_episodes(current_db, episodes, all_frames, git_commit, candidate_margin)

        # Step 5: Evaluate on fixed holdout seeds — same every round
        print(f"\n  Step 5: Holdout eval (collection: {n_landed}/{n_total} = {new_rate:.0%})")
        eval_margins = [0.001, margin_mean, candidate_margin]
        eval_margins = sorted(set(np.clip(eval_margins, 0.001, 1.0)))
        eval_results = evaluate_model_seeds(
            env, live_model, eval_margins, HOLDOUT_SEEDS,
        )

        # Check baseline performance
        baseline_lands = eval_results.get(0.001, (0, 30))[0]
        candidate_lands = eval_results.get(
            round(candidate_margin, 3),
            eval_results.get(candidate_margin, (0, 30))
        )
        if isinstance(candidate_lands, tuple):
            candidate_lands = candidate_lands[0]

        # Step 6: Add to archive
        archive_db.copy_from(current_db)
        print(f"\n  Archive updated: {archive_db.count_episodes()} episodes, "
              f"{archive_db.count_frames()} frames")

        # Step 7-8: Decide whether to keep higher margin
        improved = new_rate >= 0.5 and candidate_lands >= baseline_lands
        round_time = time.time() - t_round

        if improved:
            margin_mean = candidate_margin
            print(f"\n  IMPROVED! Keeping margin_mean={margin_mean:.3f}")
        else:
            print(f"\n  No improvement. Staying at margin_mean={margin_mean:.3f}")

        print(f"  Round {round_num} took {round_time:.0f}s")

        # Save checkpoint after each round
        ckpt_path = os.path.join(args.run_dir, f"dagger_round{round_num}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean.numpy(),
            "x_std": x_std.numpy(),
            "epochs": checkpoint.get("epochs", 0) + round_num * args.train_epochs,
            "n_frames": archive_db.count_frames(),
            "T": T,
            "margin_mean": margin_mean,
            "round": round_num,
            "hidden": args.hidden,
            "n_blocks": args.n_blocks,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final summary ────────────────────────────────────────────────────
    env.close()
    print(f"\n{'='*60}")
    print(f"DAGGER COMPLETE")
    print(f"{'='*60}")
    print(f"  Final margin_mean: {margin_mean:.3f}")
    print(f"  Archive: {archive_db.count_episodes()} episodes, "
          f"{archive_db.count_frames()} frames")
    print(f"  Checkpoints: dagger_round1.pt .. dagger_round{n_rounds}.pt")

    # Final eval sweep
    print(f"\n  Final evaluation sweep:")
    final_model = _LiveModel(model, x_mean, x_std, T=T)
    env_final_id = f"LL-dagger-final-{os.getpid()}"
    try:
        gym.register(id=env_final_id, entry_point="lunar_lander:LunarLander",
                     max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make(env_final_id, render_mode=None, continuous=True)
    final_margins = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    evaluate_model_seeds(env, final_model, final_margins, HOLDOUT_SEEDS)
    env.close()

    archive_db.close()
    current_db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
