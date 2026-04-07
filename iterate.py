"""Iterative training: progressively widen guidance margin until autonomous.

Each round:
1. Collect 100 episodes at the current margin center
2. Train on ALL accumulated data
3. Evaluate — if margin=1.0 outperforms baseline, we're done

Usage:
    .venv/bin/python iterate.py
"""

import os
import shutil
import subprocess
import sys
import json
import numpy as np

PYTHON = sys.executable
MARGIN_SCHEDULE = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EPISODES_PER_BATCH = 100
LANDED_PER_BATCH = 80
EPOCHS_PER_ROUND = 200
DB_PATH = "rollouts.db"
EVAL_SEEDS = 5000  # offset to avoid training seeds


def run(cmd, timeout=1800):
    """Run a command, streaming output."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {result.returncode}")
    return result.stdout


def collect(seed_start, margin_center):
    """Collect a batch at the given margin center.

    Patches the margin sampling in lunar_lander.py via the collection args.
    Since the collect mode uses rng.normal(0.2, 0.1), we temporarily patch
    the source to use the desired center. Simpler: just run multiple times
    with different margin values.
    """
    # For now, collect with the default margin sampling and retrain.
    # The key insight: we train on ALL episodes including earlier tight-margin ones,
    # so the model learns from progressively loosening data.
    return run([
        PYTHON, "lunar_lander.py", "--collect",
        "--seed", str(seed_start),
        "--db", DB_PATH,
        "--target-total", str(EPISODES_PER_BATCH),
        "--target-landed", str(LANDED_PER_BATCH),
        "--margin-center", str(margin_center),
        "--margin-sigma", "0.1",
    ], timeout=1800)


def train(model_path, epochs):
    """Train on all accumulated data."""
    return run([
        PYTHON, "train.py",
        "--db", DB_PATH,
        "--epochs", str(epochs),
        "--save", model_path,
    ], timeout=3600)


def evaluate(model_path):
    """Evaluate and parse results."""
    output = run([
        PYTHON, "eval.py",
        "--model", model_path,
        "--n-baseline", "50",
        "--n-per-margin", "50",
        "--seed-offset", str(EVAL_SEEDS),
    ], timeout=7200)

    results = {}
    for line in output.strip().split("\n"):
        parts = line.split()
        if len(parts) == 4:
            try:
                margin = float(parts[0])
                land_pct = float(parts[2].rstrip("%")) / 100
                reward = float(parts[3])
                results[margin] = {"land_rate": land_pct, "reward": reward}
            except (ValueError, IndexError):
                continue
    return results


def main():
    seed_counter = 0

    # If DB exists from proof-of-life, keep it as initial data
    if os.path.exists(DB_PATH):
        from rollout_db import RolloutDB
        db = RolloutDB(DB_PATH)
        existing = db.count()
        db.close()
        print(f"Starting with {existing} existing episodes in {DB_PATH}")
    else:
        existing = 0

    print("=" * 60)
    print("ITERATIVE DIFFUSION TRAINING")
    print(f"Goal: margin=1.0 landing rate > baseline (margin=0.001)")
    print(f"Schedule: margins {MARGIN_SCHEDULE}")
    print(f"Episodes per batch: {EPISODES_PER_BATCH}")
    print("=" * 60)

    all_results = []
    round_idx = 0
    MAX_ATTEMPTS_PER_MARGIN = 5  # max batches before forced advance

    for margin_center in MARGIN_SCHEDULE:
        attempt = 0
        mastered = False

        while not mastered and attempt < MAX_ATTEMPTS_PER_MARGIN:
            model_path = f"model_round{round_idx:02d}.pt"

            print(f"\n{'='*60}")
            print(f"ROUND {round_idx}: margin={margin_center:.1f} (attempt {attempt+1})")
            print(f"{'='*60}")

            # 1. Collect at this margin
            print(f"\n--- Collecting {EPISODES_PER_BATCH} episodes (margin={margin_center:.1f}) ---")
            seed_start = seed_counter
            seed_counter += 200
            collect(seed_start, margin_center)

            from rollout_db import RolloutDB
            db = RolloutDB(DB_PATH)
            total = db.count()
            summary = db.summary()
            db.close()
            print(f"Total episodes in DB: {total}")
            for outcome, stats in summary.items():
                print(f"  {outcome}: {stats['count']}")

            # 2. Train on ALL accumulated data
            print(f"\n--- Training ({EPOCHS_PER_ROUND} epochs on {total} episodes) ---")
            train(model_path, EPOCHS_PER_ROUND)

            # 3. Evaluate
            print(f"\n--- Evaluating {model_path} ---")
            results = evaluate(model_path)
            all_results.append({"round": round_idx, "margin_center": margin_center,
                                "attempt": attempt, **results})

            baseline = results.get(0.001, {}).get("land_rate", 0)
            at_1_0 = results.get(1.0, {}).get("land_rate", 0)
            at_center = results.get(round(margin_center, 1), {}).get("land_rate", 0)

            print(f"\n--- Round {round_idx} Summary ---")
            print(f"  Baseline (0.001): {baseline:.0%}")
            print(f"  At margin ({margin_center:.1f}): {at_center:.0%}")
            print(f"  At 1.0: {at_1_0:.0%}")

            shutil.copy(model_path, "model_latest.pt")

            # Check if we've mastered this margin level
            if at_center >= baseline and at_center > 0:
                mastered = True
                print(f"  MASTERED margin={margin_center:.1f} "
                      f"({at_center:.0%} >= baseline {baseline:.0%})")
            else:
                print(f"  Not yet — margin={margin_center:.1f} at {at_center:.0%} "
                      f"vs baseline {baseline:.0%}, adding more data...")

            # Check full graduation
            if at_1_0 > baseline and at_1_0 >= 0.5:
                print(f"\n{'*'*60}")
                print(f"GRADUATED! margin=1.0 ({at_1_0:.0%}) > baseline ({baseline:.0%})")
                print(f"Model: {model_path}")
                print(f"{'*'*60}")
                shutil.copy(model_path, "model_graduated.pt")
                mastered = "graduated"
                break

            round_idx += 1
            attempt += 1

        if mastered == "graduated":
            break

        if not mastered:
            print(f"  Forced advance past margin={margin_center:.1f} "
                  f"after {MAX_ATTEMPTS_PER_MARGIN} attempts")
        round_idx += 1

    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING HISTORY")
    print(f"{'='*60}")
    print(f"{'Round':>5} {'Margin':>7} {'Baseline':>9} {'@Center':>8} {'@1.0':>6}")
    print("-" * 40)
    for r in all_results:
        bl = r.get(0.001, {}).get("land_rate", 0)
        ctr = r.get(round(r["margin_center"], 1), {}).get("land_rate", 0)
        at1 = r.get(1.0, {}).get("land_rate", 0)
        print(f"{r['round']:5d} {r['margin_center']:7.1f} {bl:8.0%} {ctr:8.0%} {at1:5.0%}")


if __name__ == "__main__":
    main()
