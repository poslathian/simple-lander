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
def read_source_files():
    """Read all source files needed on Modal containers."""
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

# Training runs via subprocess (modal_train_gpu.py) to avoid Modal app entanglement


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


def _get_rollout_functions():
    """Lazy import to avoid triggering rollout image build at module load."""
    from modal_rollout import (
        app as rollout_app,
        solve_kto_batch,
        rollout_with_cache,
    )
    return rollout_app, solve_kto_batch, rollout_with_cache


def presolve_kto_pool(source_files, seeds, _solve_fn, _rollout_app, chunk_size=100):
    """Pre-solve KTO plans in chunks to avoid Modal heartbeat timeout."""
    print(f"  Pre-solving KTO plans for {len(seeds)} seeds...", flush=True)
    t0 = time.time()
    pool = {}
    chunks = [seeds[i:i+chunk_size] for i in range(0, len(seeds), chunk_size)]
    for ci, chunk in enumerate(chunks):
        batches = [chunk[i:i+20] for i in range(0, len(chunk), 20)]
        with _rollout_app.run():
            for batch_result in _solve_fn.starmap(
                [(source_files, batch) for batch in batches]
            ):
                pool.update(batch_result)
        print(f"    chunk {ci+1}/{len(chunks)}: {len(pool)} plans ({time.time()-t0:.0f}s)",
              flush=True)
    print(f"  Pool complete: {len(pool)} plans in {time.time()-t0:.0f}s")
    return pool


def collect_from_pool(ckpt_bytes, plan_pool, n_landed, n_failed,
                      margin_mean, hidden, n_blocks, source_files,
                      _rollout_fn=None):
    """Collect episodes using pre-solved KTO plan pool. No KTO solve needed."""
    # Pick random seeds from pool
    available = list(plan_pool.keys())
    np.random.shuffle(available)
    est_needed = int((n_landed + n_failed) / 0.50) + 40
    seeds = available[:min(est_needed, len(available))]

    print(f"    Rollouts ({len(seeds)} seeds from pool)...", end="", flush=True)
    t0 = time.time()
    seed_plan_pairs = [(s, plan_pool[s]) for s in seeds]
    rollout_batches = [seed_plan_pairs[i:i+10]
                       for i in range(0, len(seed_plan_pairs), 10)]

    all_results = []
    landed = 0
    failed = 0
    for batch_result in _rollout_fn.starmap([
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

    print(f" {time.time()-t0:.0f}s  ({landed} landed, {failed} failed)")
    return all_results


def evaluate_from_pool(ckpt_bytes, plan_pool, margins, holdout_seeds,
                       hidden, n_blocks, source_files, _rollout_fn=None):
    """Evaluate at multiple margins using pre-solved plans."""
    seed_plan_pairs = [(s, plan_pool[s]) for s in holdout_seeds if s in plan_pool]

    results = {}
    for batch_result in _rollout_fn.starmap([
        (source_files, ckpt_bytes, pickle.dumps(seed_plan_pairs), m, 1, hidden, n_blocks)
        for m in margins
    ]):
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

    margin_increment = 0.05
    POOL_SIZE = 500
    pool_seeds = list(range(args.seed_offset, args.seed_offset + POOL_SIZE))

    # Pre-solve KTO plans for entire seed pool + holdout seeds (one-time cost)
    rollout_app, solve_fn, rollout_fn = _get_rollout_functions()
    print(f"\n  Pre-solving KTO pool ({POOL_SIZE} + {len(HOLDOUT_SEEDS)} holdout)...")
    plan_pool = presolve_kto_pool(
        source_files, pool_seeds + HOLDOUT_SEEDS, solve_fn, rollout_app,
    )
    print(f"  Pool: {len(plan_pool)} plans cached")

    # Initial collection (skip if resuming)
    if not resuming:
        print(f"\n{'='*60}")
        print(f"INITIAL COLLECTION: margin={margin_mean:.3f}")
        print(f"{'='*60}")

        with open(ckpt_path, "rb") as f:
            ckpt_bytes = f.read()

        with rollout_app.run():
            results = collect_from_pool(
                ckpt_bytes, plan_pool,
                n_landed=args.target_landed, n_failed=args.target_failed,
                margin_mean=margin_mean,
                hidden=args.hidden, n_blocks=args.n_blocks,
                source_files=source_files, _rollout_fn=rollout_fn,
            )

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

        # Step 3: GPU training via subprocess
        print(f"\n  Training {args.train_epochs} epochs on GPU...")
        frame_tmp = os.path.join(args.run_dir, "_train_frames.pkl")
        with open(frame_tmp, "wb") as f:
            pickle.dump(frame_data, f)

        new_ckpt_path = os.path.join(args.run_dir, f"dagger_round{round_num}.pt")
        train_cmd = [
            sys.executable, "-u", "modal_train_gpu.py",
            "--checkpoint", ckpt_path,
            "--frames", frame_tmp,
            "--output", new_ckpt_path,
            "--epochs", str(args.train_epochs),
            "--lr", str(args.lr),
            "--batch-size", str(args.batch_size),
            "--hidden", str(args.hidden),
            "--n-blocks", str(args.n_blocks),
            "--T", str(T),
        ]
        t_train = time.time()
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")
        if result.returncode != 0:
            print(f"  TRAIN ERROR: {result.stderr[-500:]}")
        print(f"  Training took {time.time()-t_train:.0f}s")

        ckpt_path = new_ckpt_path
        with open(ckpt_path, "rb") as f:
            new_ckpt_bytes = f.read()
        os.unlink(frame_tmp)
        print(f"  Saved: {ckpt_path}")

        # Step 4-5: Collect + eval from cached pool (no KTO solve!)
        candidate_margin = np.clip(margin_mean + margin_increment, 0.001, 1.0)
        print(f"\n  Collecting at candidate margin={candidate_margin:.3f}...")
        current_db.clear()

        with rollout_app.run():
            new_results = collect_from_pool(
                new_ckpt_bytes, plan_pool,
                n_landed=args.target_landed, n_failed=args.target_failed,
                margin_mean=candidate_margin,
                hidden=args.hidden, n_blocks=args.n_blocks,
                source_files=source_files, _rollout_fn=rollout_fn,
            )

            current_db.store_remote_results(new_results, git_commit)
            n_landed = sum(1 for r in new_results if r["landed"])
            n_total = len(new_results)
            new_rate = n_landed / n_total if n_total else 0

            # Holdout eval
            print(f"\n  Holdout eval (collection: {n_landed}/{n_total} = {new_rate:.0%}):")
            eval_margins = sorted(set([0.001, margin_mean, candidate_margin]))
            eval_results = evaluate_from_pool(
                new_ckpt_bytes, plan_pool, eval_margins, HOLDOUT_SEEDS,
                hidden=args.hidden, n_blocks=args.n_blocks,
                source_files=source_files, _rollout_fn=rollout_fn,
            )

        baseline_lands = eval_results.get(0.001, (0, 50))[0]
        candidate_key = min(eval_results.keys(),
                            key=lambda m: abs(m - candidate_margin))
        candidate_lands = eval_results[candidate_key][0]

        # Step 6: Add to archive
        archive_db.copy_from(current_db)
        print(f"\n  Archive: {archive_db.count_episodes()} eps, "
              f"{archive_db.count_frames()} frames")

        # Step 7-8: Advance if collection >= 50% AND candidate >= 70% of baseline
        baseline_threshold = int(baseline_lands * 0.70)
        improved = new_rate >= 0.5 and candidate_lands >= baseline_threshold
        if improved:
            margin_mean = candidate_margin
            print(f"  IMPROVED! margin → {margin_mean:.3f} "
                  f"(candidate {candidate_lands} >= 70% of baseline {baseline_lands})")
        else:
            print(f"  No improvement. margin stays at {margin_mean:.3f} "
                  f"(candidate {candidate_lands} vs 70%×baseline={baseline_threshold})")

        elapsed = time.time() - t_round
        print(f"  Round took {elapsed:.0f}s")

    # Final eval sweep
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION")
    print(f"{'='*60}")
    with open(ckpt_path, "rb") as f:
        final_bytes = f.read()
    with rollout_app.run():
        evaluate_from_pool(
            final_bytes, plan_pool,
            [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
            HOLDOUT_SEEDS,
            hidden=args.hidden, n_blocks=args.n_blocks,
            source_files=source_files, _rollout_fn=rollout_fn,
        )

    archive_db.close()
    current_db.close()

    print(f"\n{'='*60}")
    print(f"COMPLETE: final margin={margin_mean:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
