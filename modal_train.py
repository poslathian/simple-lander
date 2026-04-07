"""GPU-accelerated training on Modal. Reproduces local results faster.

Usage:
    .venv/bin/python modal_train.py [--epochs 2000] [--lr 5e-4] [--db training.db]

This script:
  1. Uploads training.db and model code to Modal
  2. Trains on a T4 GPU
  3. Downloads position_model.pt
  4. Runs local eval to validate results match
"""

import argparse
import os
import sys
import tempfile

import modal

app = modal.App("position-diffusion-train")

# ── Modal image: minimal, just what training needs ────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy")
)

# ── Remote training function ──────────────────────────────────────────────

@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
)
def train_remote(
    model_py: bytes,
    train_py: bytes,
    db_bytes: bytes,
    epochs: int,
    lr: float,
    batch_size: int,
    T: int,
) -> bytes:
    """Train on GPU, return checkpoint bytes."""
    import subprocess
    import os

    workdir = "/root/train"
    os.makedirs(workdir, exist_ok=True)

    # Write files
    with open(f"{workdir}/model.py", "wb") as f:
        f.write(model_py)
    with open(f"{workdir}/train_inner.py", "wb") as f:
        f.write(train_py)
    with open(f"{workdir}/training.db", "wb") as f:
        f.write(db_bytes)

    os.chdir(workdir)

    # Run training
    result = subprocess.run(
        [
            sys.executable, "-u", "train_inner.py",
            "--db", "training.db",
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--save", "position_model.pt",
            "--T", str(T),
            "--eval-interval", "99999",  # skip eval on remote (no env available)
        ],
        capture_output=True,
        text=True,
        timeout=6000,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr[-500:]}")

    with open(f"{workdir}/position_model.pt", "rb") as f:
        return f.read()


# ── Local entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train position diffusion model on Modal GPU")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--db", default="training.db")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--output", default="position_model.pt")
    parser.add_argument("--skip-eval", action="store_true", help="Skip local eval after training")
    args = parser.parse_args()

    # Check DB exists
    if not os.path.exists(args.db):
        print(f"ERROR: {args.db} not found. Run collect.py first.")
        return 1

    # Read files to upload
    with open("model.py", "rb") as f:
        model_py = f.read()

    # Build a standalone train script (no eval, no imports that need lunar_lander)
    train_inner = _build_train_inner()

    with open(args.db, "rb") as f:
        db_bytes = f.read()

    print(f"Uploading to Modal:", flush=True)
    print(f"  model.py: {len(model_py)} bytes")
    print(f"  train_inner.py: {len(train_inner)} bytes")
    print(f"  {args.db}: {len(db_bytes)} bytes")
    print(f"\nTraining config: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}, T={args.T}")
    sys.stdout.flush()

    # Run on Modal
    with app.run():
        print("\nStarting GPU training on Modal...", flush=True)
        checkpoint_bytes = train_remote.remote(
            model_py=model_py,
            train_py=train_inner,
            db_bytes=db_bytes,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            T=args.T,
        )

    # Save checkpoint
    with open(args.output, "wb") as f:
        f.write(checkpoint_bytes)
    print(f"\nSaved checkpoint to {args.output} ({len(checkpoint_bytes)} bytes)")

    # Local eval
    if not args.skip_eval:
        print("\nRunning local eval to validate...")
        sys.stdout.flush()
        import subprocess
        ret = subprocess.run(
            [sys.executable, "-u", "eval.py", "--model", args.output, "--episodes", "50"],
        )
        return ret.returncode

    return 0


def _build_train_inner():
    """Build a standalone training script that doesn't import lunar_lander/solver."""
    return b'''"""Standalone training script for Modal (no env dependencies)."""

import sqlite3
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import DiffusionMLP, CosineSchedule, X_DIM, COND_DIM, STATE_DIM, CFG_START, CFG_END

CFG_DROPOUT_RATE = 0.5


class PositionDataset(Dataset):
    def __init__(self, db_path="training.db"):
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT cond, target_cps, outcome FROM frames").fetchall()
        conn.close()

        self.conds = []
        self.targets = []
        self.outcomes = []

        for cond_blob, target_blob, outcome in rows:
            self.conds.append(np.frombuffer(cond_blob, dtype=np.float32).copy())
            self.targets.append(np.frombuffer(target_blob, dtype=np.float32).copy())
            self.outcomes.append(outcome)

        self.conds = np.array(self.conds, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.outcomes = np.array(self.outcomes, dtype=np.float32)
        print(f"Loaded {len(self.conds)} frames. Outcomes: +1={np.sum(self.outcomes>0)}, -1={np.sum(self.outcomes<0)}")

    def __len__(self):
        return len(self.conds)

    def __getitem__(self, idx):
        cond = np.zeros(COND_DIM, dtype=np.float32)
        cond[:STATE_DIM] = self.conds[idx]
        cond[CFG_START] = self.outcomes[idx]
        if np.random.random() < CFG_DROPOUT_RATE:
            cond[CFG_START:CFG_END] = 0.0
        return (
            torch.tensor(cond, dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="training.db")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save", default="position_model.pt")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=99999)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PositionDataset(args.db)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=False, num_workers=2, pin_memory=True)

    model = DiffusionMLP().to(device)
    schedule = CosineSchedule(T=args.T)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    x_mean = torch.tensor(dataset.targets.mean(axis=0), dtype=torch.float32).to(device)
    x_std = torch.tensor(dataset.targets.std(axis=0).clip(1e-6), dtype=torch.float32).to(device)

    # Precompute alpha_bar on device
    alpha_bar_all = torch.tensor(schedule.alpha_bar, dtype=torch.float32, device=device)

    import time
    t0 = time.time()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0

        for cond, target in loader:
            cond = cond.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            B = cond.shape[0]

            x0 = (target - x_mean) / x_std
            t = torch.randint(1, schedule.T + 1, (B,), device=device)
            alpha_bar = alpha_bar_all[t].unsqueeze(1)

            noise = torch.randn_like(x0)
            x_noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

            eps_pred = model(x_noisy, cond, t)
            loss = nn.functional.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0

        if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:5d}/{args.epochs}  loss={avg_loss:.6f}  time={elapsed:.0f}s")

    # Save (move to CPU for portability)
    model.cpu()
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean.cpu().numpy(),
        "x_std": x_std.cpu().numpy(),
        "epochs": args.epochs,
        "n_frames": len(dataset),
        "T": args.T,
    }, args.save)
    print(f"\\nSaved to {args.save} ({len(dataset)} frames, {args.epochs} epochs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    sys.exit(main())
