"""Modal-based iterative training: progressively widen guidance margin.

Collects batches of KTO episodes at increasing guidance margins,
trains the diffusion model on each batch, and evaluates until
margin=1.0 outperforms the KTO baseline (margin=0.001).

Usage:
    .venv/bin/modal run modal_train.py
"""

import modal

app = modal.App("lunar-lander-diffusion")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("swig", "build-essential")
    .pip_install(
        "numpy>=2.4", "scipy>=1.17", "torch",
        "gymnasium[box2d]", "pygame", "Pillow",
        "drake>=1.51",
    )
    .copy_local_file("lunar_lander.py")
    .copy_local_file("solver.py")
    .copy_local_file("solver.pyi")
    .copy_local_file("diffusion_controller.py")
    .copy_local_file("guidance_controller.py")
    .copy_local_file("model.py")
    .copy_local_file("thrust_spline.py")
    .copy_local_file("thrust_spline.pyi")
    .copy_local_file("rollout_db.py")
    .copy_local_file("train.py")
    .copy_local_file("eval.py")
)

vol = modal.Volume.from_name("lunar-lander-training", create_if_missing=True)
VOL_PATH = "/data"


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={VOL_PATH: vol},
)
def collect_batch(
    seed_start: int,
    target_total: int,
    target_landed: int,
    margin_center: float,
    margin_sigma: float,
    db_name: str,
):
    """Collect a batch of KTO episodes on Modal."""
    import subprocess
    import shutil

    db_path = f"{VOL_PATH}/{db_name}"

    result = subprocess.run(
        [
            "python", "lunar_lander.py", "--collect",
            "--seed", str(seed_start),
            "--db", db_path,
            "--target-total", str(target_total),
            "--target-landed", str(target_landed),
        ],
        capture_output=True, text=True, timeout=1800,
        env={
            "PATH": "/usr/bin:/usr/local/bin",
            "HOME": "/root",
            "MARGIN_CENTER": str(margin_center),
            "MARGIN_SIGMA": str(margin_sigma),
        },
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-1000:]}")
        raise RuntimeError(f"Collection failed: {result.returncode}")

    vol.commit()
    return db_path


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={VOL_PATH: vol},
)
def train_model(db_name: str, epochs: int, model_name: str, prev_model: str = None):
    """Train diffusion model on collected data."""
    import subprocess

    db_path = f"{VOL_PATH}/{db_name}"
    model_path = f"{VOL_PATH}/{model_name}"

    cmd = [
        "python", "train.py",
        "--db", db_path,
        "--epochs", str(epochs),
        "--save", model_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-1000:]}")
        raise RuntimeError(f"Training failed: {result.returncode}")

    vol.commit()
    return model_path


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={VOL_PATH: vol},
)
def evaluate_model(model_name: str, n_baseline: int = 50, n_per_margin: int = 10):
    """Evaluate model across guidance margin strengths."""
    import subprocess

    model_path = f"{VOL_PATH}/{model_name}"

    result = subprocess.run(
        [
            "python", "eval.py",
            "--model", model_path,
            "--n-baseline", str(n_baseline),
            "--n-per-margin", str(n_per_margin),
            "--seed-offset", "5000",
        ],
        capture_output=True, text=True, timeout=1200,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-1000:]}")

    # Parse results from output
    lines = result.stdout.strip().split("\n")
    results = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 4:
            try:
                margin = float(parts[0])
                land_pct = parts[2].rstrip("%")
                reward = float(parts[3])
                results[margin] = {
                    "land_rate": float(land_pct) / 100,
                    "reward": reward,
                }
            except (ValueError, IndexError):
                continue
    return results


@app.local_entrypoint()
def main():
    """Iterative training loop: progressively widen guidance margin."""
    import json

    # Training schedule: margin centers to progress through
    margin_schedule = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    episodes_per_batch = 100
    landed_per_batch = 80
    epochs_per_round = 200
    seed_counter = 0

    print("=" * 60)
    print("ITERATIVE DIFFUSION TRAINING")
    print("Goal: outperform KTO baseline at margin=1.0")
    print("=" * 60)

    for round_idx, margin_center in enumerate(margin_schedule):
        round_name = f"round_{round_idx:02d}_margin_{margin_center:.1f}"
        db_name = f"{round_name}.db"
        model_name = f"{round_name}.pt"

        print(f"\n{'='*60}")
        print(f"Round {round_idx}: margin_center={margin_center:.1f}")
        print(f"{'='*60}")

        # 1. Collect episodes at this margin
        print(f"\n[1/3] Collecting {episodes_per_batch} episodes (>={landed_per_batch} landed)...")
        collect_batch.remote(
            seed_start=seed_counter,
            target_total=episodes_per_batch,
            target_landed=landed_per_batch,
            margin_center=margin_center,
            margin_sigma=0.1,
            db_name=db_name,
        )
        seed_counter += 200  # skip ahead to avoid seed overlap

        # 2. Train on this batch
        print(f"\n[2/3] Training for {epochs_per_round} epochs...")
        train_model.remote(
            db_name=db_name,
            epochs=epochs_per_round,
            model_name=model_name,
        )

        # 3. Evaluate
        print(f"\n[3/3] Evaluating...")
        results = evaluate_model.remote(
            model_name=model_name,
            n_baseline=50,
            n_per_margin=10,
        )

        # Check if we've graduated
        baseline_rate = results.get(0.001, {}).get("land_rate", 0)
        margin_1_rate = results.get(1.0, {}).get("land_rate", 0)

        print(f"\nResults: baseline={baseline_rate:.0%}, margin_1.0={margin_1_rate:.0%}")

        if margin_1_rate > baseline_rate and margin_1_rate >= 0.6:
            print(f"\n{'='*60}")
            print(f"GRADUATED at round {round_idx}!")
            print(f"margin=1.0 ({margin_1_rate:.0%}) > baseline ({baseline_rate:.0%})")
            print(f"Model: {model_name}")
            print(f"{'='*60}")
            break
        else:
            print(f"Not yet — continuing to next margin...")

    print("\nDone.")
