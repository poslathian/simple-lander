"""Modal-based iterative training pipeline.

Runs collection, training, and eval entirely on Modal with GPU.
Collection is parallelized across multiple containers.

Usage:
    .venv/bin/modal run modal_train.py
"""

import modal

app = modal.App("lunar-lander-diffusion")

# Persistent volume for DB, models, checkpoints
vol = modal.Volume.from_name("lunar-lander-data", create_if_missing=True)
VOL = "/data"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("swig", "build-essential")
    .pip_install(
        "numpy>=2.4", "scipy>=1.17", "torch",
        "gymnasium[box2d]", "pygame", "Pillow",
        "drake>=1.51",
    )
    .add_local_file("lunar_lander.py", "/root/lunar_lander.py")
    .add_local_file("solver.py", "/root/solver.py")
    .add_local_file("solver.pyi", "/root/solver.pyi")
    .add_local_file("diffusion_controller.py", "/root/diffusion_controller.py")
    .add_local_file("guidance_controller.py", "/root/guidance_controller.py")
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("thrust_spline.py", "/root/thrust_spline.py")
    .add_local_file("thrust_spline.pyi", "/root/thrust_spline.pyi")
    .add_local_file("rollout_db.py", "/root/rollout_db.py")
    .add_local_file("train.py", "/root/train.py")
    .add_local_file("eval.py", "/root/eval.py")
)


@app.function(image=image, timeout=1200, volumes={VOL: vol}, cpu=4)
def collect_batch(
    seed_start: int,
    n_episodes: int,
    n_landed: int,
    margin_center: float,
    margin_sigma: float,
    db_path: str,
) -> str:
    """Collect a batch of KTO episodes."""
    import subprocess
    result = subprocess.run(
        [
            "python", "/root/lunar_lander.py", "--collect",
            "--seed", str(seed_start),
            "--db", db_path,
            "--target-total", str(n_episodes),
            "--target-landed", str(n_landed),
            "--margin-center", str(margin_center),
            "--margin-sigma", str(margin_sigma),
        ],
        capture_output=True, text=True, timeout=1000,
        cwd="/root",
    )
    vol.commit()
    # Return last few lines as status
    lines = result.stdout.strip().split("\n")
    return "\n".join(lines[-5:])


@app.function(image=image, gpu="T4", timeout=3600, volumes={VOL: vol})
def train_model(db_path: str, epochs: int, model_path: str) -> str:
    """Train on all accumulated data with GPU."""
    import subprocess
    result = subprocess.run(
        [
            "python", "/root/train.py",
            "--db", db_path,
            "--epochs", str(epochs),
            "--save", model_path,
        ],
        capture_output=True, text=True, timeout=3000,
        cwd="/root",
    )
    vol.commit()
    lines = result.stdout.strip().split("\n")
    return "\n".join(lines[-5:])


@app.function(image=image, timeout=7200, volumes={VOL: vol}, cpu=4)
def run_eval(model_path: str, n_baseline: int, n_per_margin: int, seed_offset: int) -> dict:
    """Evaluate model across margin strengths."""
    import subprocess
    db_path = f"{VOL}/rollouts.db"
    result = subprocess.run(
        [
            "python", "/root/eval.py",
            "--model", model_path,
            "--db", db_path,
            "--n-baseline", str(n_baseline),
            "--n-per-margin", str(n_per_margin),
            "--seed-offset", str(seed_offset),
        ],
        capture_output=True, text=True, timeout=7000,
        cwd="/root",
    )
    # Parse results table (new format: margin  N  land  land%  reward)
    results = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            try:
                margin = float(parts[0])
                land_frac = parts[3].rstrip("%")
                land_pct = float(land_frac) / 100
                reward = float(parts[4])
                results[str(margin)] = {"land_rate": land_pct, "reward": reward}
            except (ValueError, IndexError):
                continue
        # Capture MSE and loss lines
        if "Diffusion MSE:" in line:
            results["_diff_mse"] = float(line.split(":")[-1].strip())
        if "Train loss:" in line:
            results["_train_loss"] = float(line.split(":")[1].split("(")[0].strip())
    print(result.stdout[-3000:])
    return results


@app.function(image=image, timeout=600, volumes={VOL: vol})
def merge_dbs(source_paths: list[str], dest_path: str) -> int:
    """Merge multiple collection DBs into one."""
    import sqlite3
    dest = sqlite3.connect(dest_path)
    for src_path in source_paths:
        src = sqlite3.connect(src_path)
        # Copy all rows from source to dest
        for row in src.execute("SELECT * FROM episodes"):
            cols = [d[0] for d in src.execute("SELECT * FROM episodes LIMIT 1").description]
            placeholders = ",".join(["?"] * (len(cols) - 1))  # skip id
            dest.execute(
                f"INSERT INTO episodes ({','.join(cols[1:])}) VALUES ({placeholders})",
                row[1:],
            )
        src.close()
    dest.commit()
    count = dest.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    dest.close()
    vol.commit()
    return count


@app.local_entrypoint()
def main():
    import json

    MARGIN_SCHEDULE = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    EPISODES_PER_BATCH = 100
    LANDED_PER_BATCH = 80
    EPOCHS = 200
    N_PARALLEL_COLLECTORS = 4  # parallelize collection
    EPISODES_PER_COLLECTOR = EPISODES_PER_BATCH // N_PARALLEL_COLLECTORS
    LANDED_PER_COLLECTOR = LANDED_PER_BATCH // N_PARALLEL_COLLECTORS
    MAX_ATTEMPTS = 5
    DB = f"{VOL}/rollouts.db"

    seed_counter = 0
    round_idx = 0

    print("=" * 60)
    print("MODAL ITERATIVE DIFFUSION TRAINING")
    print(f"Goal: margin=1.0 > baseline (margin=0.001)")
    print(f"Parallel collectors: {N_PARALLEL_COLLECTORS}")
    print("=" * 60)

    for margin_center in MARGIN_SCHEDULE:
        attempt = 0
        mastered = False

        while not mastered and attempt < MAX_ATTEMPTS:
            model_path = f"{VOL}/model_round{round_idx:02d}.pt"

            print(f"\n{'='*60}")
            print(f"ROUND {round_idx}: margin={margin_center:.1f} (attempt {attempt+1})")
            print(f"{'='*60}")

            # 1. Parallel collection
            print(f"\n[1/3] Collecting {EPISODES_PER_BATCH} episodes across {N_PARALLEL_COLLECTORS} workers...")
            shard_dbs = []
            futures = []
            for i in range(N_PARALLEL_COLLECTORS):
                shard_db = f"{VOL}/shard_{round_idx:02d}_{i}.db"
                shard_dbs.append(shard_db)
                futures.append(collect_batch.spawn(
                    seed_start=seed_counter + i * 200,
                    n_episodes=EPISODES_PER_COLLECTOR,
                    n_landed=LANDED_PER_COLLECTOR,
                    margin_center=margin_center,
                    margin_sigma=0.1,
                    db_path=shard_db,
                ))
            seed_counter += N_PARALLEL_COLLECTORS * 200

            for i, f in enumerate(futures):
                status = f.get()
                print(f"  Worker {i}: {status}")

            # Merge shards into main DB
            total = merge_dbs.remote(shard_dbs, DB)
            print(f"  Merged → {total} total episodes")

            # 2. Train on GPU
            print(f"\n[2/3] Training {EPOCHS} epochs on GPU...")
            train_status = train_model.remote(DB, EPOCHS, model_path)
            print(f"  {train_status}")

            # 3. Evaluate
            print(f"\n[3/3] Evaluating (50 per margin)...")
            results = run_eval.remote(model_path, 50, 50, 5000)

            baseline = results.get("0.001", {}).get("land_rate", 0)
            at_center = results.get(str(round(margin_center, 1)), {}).get("land_rate", 0)
            at_1_0 = results.get("1.0", {}).get("land_rate", 0)

            diff_mse = results.get("_diff_mse", float("nan"))
            train_loss = results.get("_train_loss", float("nan"))

            print(f"\n--- Round {round_idx} ---")
            print(f"  Train loss: {train_loss:.4f}  Diffusion MSE: {diff_mse:.4f}")
            print(f"  Baseline (0.001): {baseline:.0%}")
            print(f"  At margin ({margin_center:.1f}): {at_center:.0%}")
            print(f"  At 1.0: {at_1_0:.0%}")

            if at_center >= baseline and at_center > 0:
                mastered = True
                print(f"  MASTERED margin={margin_center:.1f}!")

            if at_1_0 > baseline and at_1_0 >= 0.5:
                print(f"\n{'*'*60}")
                print(f"GRADUATED! margin=1.0 ({at_1_0:.0%}) > baseline ({baseline:.0%})")
                print(f"{'*'*60}")
                return

            round_idx += 1
            attempt += 1

        if not mastered:
            print(f"  Forced advance past margin={margin_center:.1f}")
            round_idx += 1

    print("\nCompleted all margin levels.")
