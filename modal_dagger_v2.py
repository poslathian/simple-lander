"""Modal DAgger v2: GPU training + parallel cached rollouts.

Orchestrates locally, dispatches compute to Modal:
  - KTO solve: parallel CPU containers (modal_rollout.solve_kto_batch)
  - Rollouts: parallel CPU containers with cached plans (modal_rollout.rollout_with_cache)
  - Training: GPU container (T4/A10G)
  - Evaluation: parallel CPU containers (modal_rollout.rollout_with_cache at fixed margin)

Usage:
    .venv/bin/python modal_dagger_v2.py --hidden 1024 --rounds 20 --run-dir runs/D_h1024_gpu
    .venv/bin/python modal_dagger_v2.py --hidden 2048 --rounds 20 --run-dir runs/E_h2048_gpu
"""

import argparse
import io
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

from model import DiffusionMLP, CosineSchedule, DDIMSampler
from model import X_DIM, COND_DIM, STATE_DIM, CFG_START, CFG_END, N_CPS, N_CHANNELS
from modal_rollout import (
    app as rollout_app,
    read_source_files, _write_sources,
    solve_kto_batch, rollout_with_cache,
    rollout_image,
)

# Extend the rollout app with our training function
app = rollout_app

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy")
)


# ── GPU training function ────────────────────────────────────────────────

@app.function(
    image=train_image,
    gpu="T4",
    timeout=3600,
)
def train_gpu(
    model_py_bytes: bytes,
    checkpoint_bytes: bytes,
    frame_data: bytes,
    epochs: int,
    lr: float,
    batch_size: int,
    T: int,
    hidden: int,
    n_blocks: int,
    print_interval: int = 50,
) -> bytes:
    """Train on GPU, return updated checkpoint bytes."""
    import sys
    import os

    os.makedirs("/root/train", exist_ok=True)
    with open("/root/train/model.py", "wb") as f:
        f.write(model_py_bytes)
    sys.path.insert(0, "/root/train")
    os.chdir("/root/train")

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from model import DiffusionMLP, CosineSchedule
    from model import COND_DIM, STATE_DIM, CFG_START, CFG_END

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, hidden={hidden}, n_blocks={n_blocks}")

    # Load checkpoint
    with open("/tmp/ckpt.pt", "wb") as f:
        f.write(checkpoint_bytes)
    ckpt = torch.load("/tmp/ckpt.pt", weights_only=False, map_location=device)

    model = DiffusionMLP(hidden=hidden, n_blocks=n_blocks).to(device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded pre-trained weights")
    except RuntimeError:
        print("Size mismatch — training from current model state")
        # If checkpoint has matching keys from a prior round, load those
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        except Exception:
            pass

    # Load frame data
    frame_list = pickle.loads(frame_data)
    conds = np.array([f[0] for f in frame_list], dtype=np.float32)
    targets = np.array([f[1] for f in frame_list], dtype=np.float32)
    print(f"Training on {len(conds)} frames for {epochs} epochs, bs={batch_size}, lr={lr}")

    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32).to(device)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32).to(device)

    # Blend normalization stats
    new_mean = torch.tensor(targets.mean(axis=0), dtype=torch.float32).to(device)
    new_std = torch.tensor(targets.std(axis=0).clip(1e-6), dtype=torch.float32).to(device)
    x_mean = 0.7 * x_mean + 0.3 * new_mean
    x_std = 0.7 * x_std + 0.3 * new_std

    schedule = CosineSchedule(T=T)
    alpha_bar_all = torch.tensor(schedule.alpha_bar, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
            print(f"  epoch {epoch+1:>4}/{epochs}  loss={avg_loss:.6f}  "
                  f"mse={avg_mse:.4f}  {elapsed:.0f}s")

    elapsed = _time.time() - t0
    print(f"Training done in {elapsed:.0f}s")

    model.cpu()
    result = {
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean.cpu().numpy(),
        "x_std": x_std.cpu().numpy(),
        "epochs": ckpt.get("epochs", 0) + epochs,
        "n_frames": len(conds),
        "T": T,
        "hidden": hidden,
        "n_blocks": n_blocks,
    }
    buf = io.BytesIO()
    torch.save(result, buf)
    return buf.getvalue()


# ── DB (same schema as dagger_loop.py) ───────────────────────────────────

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
                cond = np.zeros(COND_DIM, dtype=np.float32)
                cond[:STATE_DIM] = inp[:STATE_DIM]
                cond[CFG_START] = float(actual_outcome)
                result.append((cond, cps))
            return result

        return _fetch(1, n_landed) + _fetch(-1, n_failed)

    def copy_from(self, other):
        for row in other.conn.execute("SELECT * FROM episodes").fetchall():
            self.conn.execute("INSERT OR IGNORE INTO episodes VALUES (?,?,?,?,?,?,?,?)", row)
        for row in other.conn.execute("SELECT * FROM frames").fetchall():
            self.conn.execute("INSERT OR IGNORE INTO frames VALUES (?,?,?,?,?,?)", row)
        self.conn.commit()

    def close(self):
        self.conn.close()


# ── Orchestration helpers ────────────────────────────────────────────────

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def collect_with_cache(source_files, ckpt_bytes, n_landed, n_failed,
                       margin_mean, seed_offset, hidden, n_blocks):
    """Two-phase collection: parallel KTO solve → parallel cached rollouts."""
    # Overestimate seeds needed
    est_total = int((n_landed + n_failed) / 0.50) + 40
    seeds = list(range(seed_offset, seed_offset + est_total))

    # Phase 1: KTO solve
    print(f"    KTO solve ({len(seeds)} seeds)...", end="", flush=True)
    t0 = time.time()
    kto_batches = [seeds[i:i+20] for i in range(0, len(seeds), 20)]
    all_plans = {}
    for batch_result in solve_kto_batch.starmap(
        [(source_files, batch) for batch in kto_batches]
    ):
        all_plans.update(batch_result)
    print(f" {time.time()-t0:.0f}s")

    # Phase 2: Rollouts with cached plans
    print(f"    Rollouts...", end="", flush=True)
    t1 = time.time()
    seed_plan_pairs = [(s, all_plans[s]) for s in seeds if s in all_plans]
    rollout_batches = [seed_plan_pairs[i:i+10]
                       for i in range(0, len(seed_plan_pairs), 10)]

    all_results = []
    landed = 0
    failed = 0
    for batch_result in rollout_with_cache.starmap([
        (source_files, ckpt_bytes, pickle.dumps(batch), margin_mean, 1, hidden, n_blocks)
        for batch in rollout_batches
    ]):
        all_results.extend(batch_result)
        for ep in batch_result:
            if ep["landed"]:
                landed += 1
            else:
                failed += 1
        if landed >= n_landed and failed >= n_failed:
            break

    print(f" {time.time()-t1:.0f}s  ({landed} landed, {failed} failed)")
    return all_results


def evaluate_with_cache(source_files, ckpt_bytes, margins, holdout_seeds,
                        hidden, n_blocks):
    """Evaluate at multiple margins using cached KTO plans."""
    # Solve KTO for holdout seeds once
    kto_batches = [holdout_seeds[i:i+20] for i in range(0, len(holdout_seeds), 20)]
    all_plans = {}
    for batch_result in solve_kto_batch.starmap(
        [(source_files, batch) for batch in kto_batches]
    ):
        all_plans.update(batch_result)

    # Run each margin as a separate batch
    results = {}
    seed_plan_pairs = [(s, all_plans[s]) for s in holdout_seeds if s in all_plans]

    for batch_result in rollout_with_cache.starmap([
        (source_files, ckpt_bytes, pickle.dumps(seed_plan_pairs), m, 1, hidden, n_blocks)
        for m in margins
    ]):
        # Each batch_result is a list of episode dicts all at the same margin
        if not batch_result:
            continue
        m = batch_result[0]["margin"]
        lands = sum(1 for r in batch_result if r["landed"])
        total = len(batch_result)
        results[m] = (lands, total)
        print(f"    margin={m:.3f}: {lands}/{total} = {lands/total:.0%}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="position_model_923760b.pt")
    parser.add_argument("--resume-checkpoint", default=None,
                        help="Resume from this checkpoint (keeps archive DB)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--train-frames", type=int, default=2000)
    parser.add_argument("--train-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--target-landed", type=int, default=80)
    parser.add_argument("--target-failed", type=int, default=20)
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--seed-offset", type=int, default=600000)
    parser.add_argument("--resume-margin", type=float, default=None)
    args = parser.parse_args()

    git_commit = _git_commit()
    T = 100
    source_files = read_source_files()
    HOLDOUT_SEEDS = list(range(90000, 90050))

    os.makedirs(args.run_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"MODAL DAGGER v2 — GPU training + cached parallel rollouts")
    print(f"{'='*60}")
    print(f"  hidden={args.hidden} blocks={args.n_blocks} "
          f"({sum(p.numel() for p in DiffusionMLP(hidden=args.hidden, n_blocks=args.n_blocks).parameters()):,} params)")
    print(f"  frames={args.train_frames} epochs={args.train_epochs} "
          f"bs={args.batch_size} lr={args.lr}")
    print(f"  collect: {args.target_landed} landed + {args.target_failed} failed")
    print(f"  run_dir={args.run_dir}")

    # DB setup
    archive_db = DaggerDB(os.path.join(args.run_dir, "archive.db"))
    current_db = DaggerDB(os.path.join(args.run_dir, "current.db"))

    # Determine starting checkpoint and margin
    resuming = args.resume_checkpoint is not None or args.resume_margin is not None
    if resuming:
        ckpt_path = args.resume_checkpoint or args.checkpoint
        margin_mean = args.resume_margin or 0.2
        current_db.clear()
        print(f"  RESUMING from {ckpt_path}, margin={margin_mean:.3f}")
        print(f"  Archive: {archive_db.count_episodes()} eps, {archive_db.count_frames()} frames")
    else:
        ckpt_path = args.checkpoint
        margin_mean = 0.2
        archive_db.clear()
        current_db.clear()
        print(f"  Fresh start from {ckpt_path}")

    seed_counter = args.seed_offset
    margin_increment = 0.05

    with app.run():
        # Initial collection (skip if resuming)
        if not resuming:
            print(f"\n{'='*60}")
            print(f"INITIAL COLLECTION: margin={margin_mean:.3f}")
            print(f"{'='*60}")

            with open(ckpt_path, "rb") as f:
                ckpt_bytes = f.read()

            results = collect_with_cache(
                source_files, ckpt_bytes,
                n_landed=args.target_landed, n_failed=args.target_failed,
                margin_mean=margin_mean, seed_offset=seed_counter,
                hidden=args.hidden, n_blocks=args.n_blocks,
            )
            seed_counter += len(results) + 50

            current_db.store_remote_results(results, git_commit)
            archive_db.copy_from(current_db)

            n_landed = sum(1 for r in results if r["landed"])
            print(f"  {n_landed}/{len(results)} landed = {n_landed/len(results):.0%}")
            print(f"  Archive: {archive_db.count_frames()} frames")

        # DAgger rounds
        print(f"\n{'='*60}")
        print(f"Starting {args.rounds} DAgger rounds")
        print(f"{'='*60}")

        with open("model.py", "rb") as f:
            model_py_bytes = f.read()

        for round_idx in range(args.rounds):
            round_num = round_idx + 1
            t_round = time.time()
            print(f"\n{'─'*60}")
            print(f"ROUND {round_num}/{args.rounds}  |  margin={margin_mean:.3f}  "
                  f"|  archive={archive_db.count_frames()} frames")
            print(f"{'─'*60}")

            # Step 1-2: Sample frames
            print(f"\n  Sampling {args.train_frames} frames...")
            frame_data = archive_db.sample_training_frames(
                args.train_frames, landed_frac=0.8,
            )
            print(f"  Got {len(frame_data)} frames")

            # Step 3: GPU training
            print(f"\n  Training {args.train_epochs} epochs on GPU...")
            with open(ckpt_path, "rb") as f:
                ckpt_bytes = f.read()

            new_ckpt_bytes = train_gpu.remote(
                model_py_bytes=model_py_bytes,
                checkpoint_bytes=ckpt_bytes,
                frame_data=pickle.dumps(frame_data),
                epochs=args.train_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                T=T,
                hidden=args.hidden,
                n_blocks=args.n_blocks,
                print_interval=max(50, args.train_epochs // 10),
            )

            # Save checkpoint
            ckpt_path = os.path.join(args.run_dir, f"dagger_round{round_num}.pt")
            with open(ckpt_path, "wb") as f:
                f.write(new_ckpt_bytes)
            print(f"  Saved: {ckpt_path}")

            # Step 4: Collect at candidate margin
            candidate_margin = np.clip(margin_mean + margin_increment, 0.001, 1.0)
            print(f"\n  Collecting at candidate margin={candidate_margin:.3f}...")
            current_db.clear()

            new_results = collect_with_cache(
                source_files, new_ckpt_bytes,
                n_landed=args.target_landed, n_failed=args.target_failed,
                margin_mean=candidate_margin, seed_offset=seed_counter,
                hidden=args.hidden, n_blocks=args.n_blocks,
            )
            seed_counter += len(new_results) + 50

            current_db.store_remote_results(new_results, git_commit)
            n_landed = sum(1 for r in new_results if r["landed"])
            n_total = len(new_results)
            new_rate = n_landed / n_total if n_total else 0

            # Step 5: Holdout eval
            print(f"\n  Holdout eval (collection: {n_landed}/{n_total} = {new_rate:.0%}):")
            eval_margins = sorted(set([0.001, margin_mean, candidate_margin]))
            eval_results = evaluate_with_cache(
                source_files, new_ckpt_bytes, eval_margins, HOLDOUT_SEEDS,
                hidden=args.hidden, n_blocks=args.n_blocks,
            )

            baseline_lands = eval_results.get(0.001, (0, 50))[0]
            candidate_key = min(eval_results.keys(),
                                key=lambda m: abs(m - candidate_margin))
            candidate_lands = eval_results[candidate_key][0]

            # Step 6: Add to archive
            archive_db.copy_from(current_db)
            print(f"\n  Archive: {archive_db.count_episodes()} eps, "
                  f"{archive_db.count_frames()} frames")

            # Step 7-8: Decide margin
            improved = new_rate >= 0.5 and candidate_lands >= baseline_lands
            if improved:
                margin_mean = candidate_margin
                print(f"  IMPROVED! margin → {margin_mean:.3f}")
            else:
                print(f"  No improvement. margin stays at {margin_mean:.3f}")

            elapsed = time.time() - t_round
            print(f"  Round took {elapsed:.0f}s")

        # Final eval sweep
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION")
        print(f"{'='*60}")
        with open(ckpt_path, "rb") as f:
            final_bytes = f.read()
        evaluate_with_cache(
            source_files, final_bytes,
            [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
            HOLDOUT_SEEDS,
            hidden=args.hidden, n_blocks=args.n_blocks,
        )

    archive_db.close()
    current_db.close()

    print(f"\n{'='*60}")
    print(f"COMPLETE: final margin={margin_mean:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
