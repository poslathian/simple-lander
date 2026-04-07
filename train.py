"""Train diffusion model on collected position B-spline data.

Loads training frames from training.db, trains DiffusionMLP with
L2 loss on noise prediction, with 50% CFG dropout on outcome dim.
"""

import sqlite3
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import DiffusionMLP, CosineSchedule, X_DIM, COND_DIM, STATE_DIM, CFG_START, CFG_END, N_CPS, N_CHANNELS


CFG_DROPOUT_RATE = 0.5


class PositionDataset(Dataset):
    """Loads all training frames from the DB into memory."""

    def __init__(self, db_path: str = "training.db"):
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT cond, target_cps, outcome FROM frames"
        ).fetchall()
        conn.close()

        self.conds = []
        self.targets = []
        self.outcomes = []

        for cond_blob, target_blob, outcome in rows:
            cond = np.frombuffer(cond_blob, dtype=np.float32).copy()
            target = np.frombuffer(target_blob, dtype=np.float32).copy()
            self.conds.append(cond)
            self.targets.append(target)
            self.outcomes.append(outcome)

        self.conds = np.array(self.conds, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.outcomes = np.array(self.outcomes, dtype=np.float32)
        print(f"Loaded {len(self.conds)} training frames from {db_path}")
        print(f"  Cond shape: {self.conds.shape}, Target shape: {self.targets.shape}")
        print(f"  Outcomes: +1={np.sum(self.outcomes > 0)}, -1={np.sum(self.outcomes < 0)}")

    def __len__(self):
        return len(self.conds)

    def __getitem__(self, idx):
        # Build full 21-dim conditioning: 20-dim state + 1-dim outcome
        cond = np.zeros(COND_DIM, dtype=np.float32)
        cond[:STATE_DIM] = self.conds[idx]
        cond[CFG_START] = self.outcomes[idx]  # +1 or -1

        # CFG dropout: zero the outcome dim with probability
        if np.random.random() < CFG_DROPOUT_RATE:
            cond[CFG_START:CFG_END] = 0.0

        return (
            torch.tensor(cond, dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def _quick_eval(model_state, x_mean, x_std, T, n_seeds=20, seed_offset=9000):
    """Quick eval: run trained model at a few margins, return summary."""
    import gymnasium as gym
    from diffusion_controller import KTODiffusionController, NoiseModel, Outcome
    from lunar_lander import LunarLander, KTOController

    class _QuickModel:
        def __init__(self, state_dict, x_mean, x_std, T):
            self.mlp = DiffusionMLP()
            self.mlp.load_state_dict(state_dict)
            self.mlp.eval()
            schedule = CosineSchedule(T=T)
            from model import DDIMSampler as DDIM
            self.sampler = DDIM(self.mlp, schedule, n_steps=10)
            self.x_mean = torch.tensor(x_mean)
            self.x_std = torch.tensor(x_std)

        def predict(self, cond, outcome, guidance_scale=2.0):
            full = np.zeros(COND_DIM, dtype=np.float32)
            full[:STATE_DIM] = cond[:STATE_DIM]
            full[CFG_START] = float(outcome)
            ct = torch.tensor(full, dtype=torch.float32).unsqueeze(0)
            x_norm = self.sampler.sample_cfg(ct, guidance_scale=guidance_scale)
            x_raw = (x_norm * self.x_std + self.x_mean).squeeze(0).numpy()
            cps = x_raw.reshape(N_CPS, N_CHANNELS)
            cps[0] = [0, 0, 0]
            return cps

    try:
        gym.register(id="LL-qeval", entry_point="lunar_lander:LunarLander", max_episode_steps=1000)
    except Exception:
        pass
    env = gym.make("LL-qeval", render_mode=None, continuous=True)
    qm = _QuickModel(model_state, x_mean, x_std, T)
    seeds = [seed_offset + i for i in range(n_seeds)]

    results = {}
    for margin in [0.001, 0.3, 0.5]:
        lands = 0
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            uw = env.unwrapped
            uw.lander.linearVelocity = (0.0, 0.0)
            uw.lander.angularVelocity = 0.0
            ctrl = KTODiffusionController(env, model=qm, target_frequency=3.0,
                                          action_horizon=1.5, outcome=Outcome.SUCCESS)
            ctrl.warm_start(time_budget=5.0)
            done, step_idx, last_inf = False, 0, -999
            spi = max(1, int(round(1.0 / (ctrl.target_frequency * 0.02))))
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
        results[margin] = lands
    env.close()
    return results


def train(
    db_path: str = "training.db",
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str = "position_model.pt",
    T: int = 100,
    eval_interval: int = 50,
):
    dataset = PositionDataset(db_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = DiffusionMLP()
    schedule = CosineSchedule(T=T)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    x_mean = torch.tensor(dataset.targets.mean(axis=0), dtype=torch.float32)
    x_std = torch.tensor(dataset.targets.std(axis=0).clip(1e-6), dtype=torch.float32)

    import time as _time
    t_start = _time.time()
    last_eval_time = t_start

    print(f"\n{'Epoch':>6} {'Loss':>10} {'m=.001':>8} {'m=.3':>8} {'m=.5':>8} {'Time':>8}")
    print("-" * 55)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for cond, target in loader:
            B = cond.shape[0]
            x0 = (target - x_mean) / x_std
            t = torch.randint(1, schedule.T + 1, (B,))
            alpha_bar = torch.tensor([schedule.get_alpha_bar(ti.item()) for ti in t]).unsqueeze(1)
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
        elapsed = _time.time() - t_start

        # Print progress + eval periodically
        do_eval = (epoch + 1) % eval_interval == 0 or epoch == 0 or epoch == epochs - 1
        if do_eval or (epoch + 1) % 10 == 0:
            if do_eval:
                model.eval()
                ev = _quick_eval(model.state_dict(), x_mean.numpy(), x_std.numpy(), T)
                model.train()
                print(f"{epoch+1:>6} {avg_loss:>10.6f} {ev[0.001]:>6}/20 {ev[0.3]:>6}/20 {ev[0.5]:>6}/20 {elapsed:>7.0f}s")
                last_eval_time = _time.time()
            else:
                print(f"{epoch+1:>6} {avg_loss:>10.6f}")

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean.numpy(),
        "x_std": x_std.numpy(),
        "epochs": epochs,
        "n_frames": len(dataset),
        "T": T,
    }, save_path)
    print(f"\nSaved to {save_path} ({len(dataset)} frames, {epochs} epochs)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="training.db")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", default="position_model.pt")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=50)
    args = parser.parse_args()

    train(args.db, args.epochs, args.batch_size, args.lr, args.save, args.T, args.eval_interval)
