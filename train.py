"""Train diffusion model on collected KTO rollouts.

Loads training frames from rollouts.db, trains DiffusionMLP with
L2 loss on fitted B-spline CPs, with 50% CFG dropout.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rollout_db import RolloutDB
from model import DiffusionMLP, CosineSchedule, X_DIM, COND_DIM, CFG_START, CFG_END


CFG_DROPOUT_RATE = 0.5


class RolloutDataset(Dataset):
    """Loads all valid training frames from the DB into memory."""

    def __init__(self, db_path: str = "rollouts.db"):
        db = RolloutDB(db_path)
        self.conds = []
        self.targets = []
        self.cfg_wps = []
        self.cfg_crashes = []

        n = db.count()
        for ep_id in range(1, n + 1):
            for cond, target_cps, cfg_wp, cfg_crash_t in db.get_training_frames(ep_id):
                self.conds.append(cond)
                self.targets.append(target_cps)
                self.cfg_wps.append(cfg_wp)
                self.cfg_crashes.append(cfg_crash_t)

        db.close()

        self.conds = np.array(self.conds, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(f"Loaded {len(self.conds)} training frames from {db_path}")

    def __len__(self):
        return len(self.conds)

    def __getitem__(self, idx):
        cond = self.conds[idx].copy()

        # CFG dropout: zero the last 12 dims with probability CFG_DROPOUT_RATE
        if np.random.random() < CFG_DROPOUT_RATE:
            cond[CFG_START:CFG_END] = 0.0

        return (
            torch.tensor(cond, dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def train(
    db_path: str = "rollouts.db",
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str = "model.pt",
):
    dataset = RolloutDataset(db_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = DiffusionMLP()
    schedule = CosineSchedule(T=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Compute normalization stats from training data
    x_mean = torch.tensor(dataset.targets.mean(axis=0), dtype=torch.float32)
    x_std = torch.tensor(dataset.targets.std(axis=0).clip(1e-6), dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for cond, target in loader:
            B = cond.shape[0]

            # Normalize targets
            x0 = (target - x_mean) / x_std

            # Sample random timestep
            t = torch.randint(1, schedule.T + 1, (B,))
            alpha_bar = torch.tensor([schedule.get_alpha_bar(ti.item()) for ti in t])
            alpha_bar = alpha_bar.unsqueeze(1)  # (B, 1)

            # Forward diffusion: add noise
            noise = torch.randn_like(x0)
            x_noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

            # Predict noise
            eps_pred = model(x_noisy, cond, t)

            # L2 loss on noise prediction
            loss = nn.functional.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.6f}")

    # Save model + normalization stats
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean.numpy(),
        "x_std": x_std.numpy(),
        "epochs": epochs,
        "n_frames": len(dataset),
    }, save_path)
    print(f"\nSaved to {save_path} ({len(dataset)} frames, {epochs} epochs)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="rollouts.db")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", default="model.pt")
    args = parser.parse_args()

    train(args.db, args.epochs, args.batch_size, args.lr, args.save)
