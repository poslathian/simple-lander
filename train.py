"""Train the 64-dim diffusion transformer locally on the cached KTO pool.

Run:
    .venv/bin/python train.py

Outputs:
    ckpt.pt           — model + dataset normalisation stats
    samples.png       — qualitative DDIM samples vs ground truth
    train_curve.png   — loss curve
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from bspline import N_CP, eval_spline
from data import (
    SHIFT_S,
    SHIFT_STEPS,
    WINDOW_LEN,
    WINDOW_S,
    WindowPairDataset,
    build_pairs,
    extract_plans,
    load_pool,
)
from diffusion import T_TRAIN, ddim_sample, make_schedule, q_sample
from model import DiffusionTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="kto_pool.pkl")
    p.add_argument("--stride", type=int, default=2,
                   help="Window stride within each plan (smaller -> more pairs).")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── Data ───────────────────────────────────────────────────────────────
    print(f"Loading cache from {args.cache}")
    pool = load_pool(args.cache)
    plans = extract_plans(pool)
    print(f"  {len(plans)} plans, building pairs (stride={args.stride})...")
    t0 = time.time()
    cond_np, tgt_np = build_pairs(plans, stride=args.stride)
    print(f"  pairs: {cond_np.shape[0]}  ({time.time()-t0:.1f}s)")

    full_ds = WindowPairDataset(cond_np, tgt_np)
    n_val = max(1, int(0.05 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False)
    print(f"  train={n_train} val={n_val}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = DiffusionTransformer(d_model=64, n_heads=4, n_blocks=args.n_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, d_model=64, blocks={args.n_blocks}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    _, _, alpha_bar = make_schedule(T_TRAIN, device=device)

    # ── Training ───────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    last_status = time.time()
    print("Training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for cond, tgt in train_loader:
            cond = cond.to(device)
            tgt = tgt.to(device)
            B = cond.shape[0]
            t = torch.randint(0, T_TRAIN, (B,), device=device)
            x_t, noise = q_sample(tgt, t, alpha_bar)
            eps_pred = model(cond, x_t, t)
            loss = ((eps_pred - noise) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())

        train_losses.append(float(np.mean(epoch_losses)))

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            v = []
            for cond, tgt in val_loader:
                cond = cond.to(device)
                tgt = tgt.to(device)
                B = cond.shape[0]
                t = torch.randint(0, T_TRAIN, (B,), device=device)
                x_t, noise = q_sample(tgt, t, alpha_bar)
                eps_pred = model(cond, x_t, t)
                v.append(((eps_pred - noise) ** 2).mean().item())
            val_losses.append(float(np.mean(v)))

        # Periodic status (>5 min runs need this so peeking works)
        now = time.time()
        if epoch == 0 or epoch == args.epochs - 1 or now - last_status > 10:
            print(
                f"  ep {epoch+1:3d}/{args.epochs}  "
                f"train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}  "
                f"({now - last_status:.1f}s since last)",
                flush=True,
            )
            last_status = now

    # ── Save ───────────────────────────────────────────────────────────────
    ckpt = {
        "state_dict": model.state_dict(),
        "ds_mean": full_ds.mean,
        "ds_std": full_ds.std,
        "args": vars(args),
    }
    torch.save(ckpt, "ckpt.pt")
    print("Saved ckpt.pt")

    # ── Loss plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE(eps_pred, noise)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Diffusion transformer training loss")
    fig.tight_layout()
    fig.savefig("train_curve.png", dpi=120)
    print("Saved train_curve.png")

    # ── Qualitative samples ────────────────────────────────────────────────
    plot_samples(model, val_ds, full_ds, device, n=6, ddim_steps=args.ddim_steps)


def plot_samples(model, val_ds, full_ds, device, n=6, ddim_steps=50):
    """For n random validation examples: draw cond window, GT future, sampled future."""
    model.eval()
    idxs = np.random.RandomState(0).choice(len(val_ds), size=n, replace=False)
    cond_batch = torch.stack([val_ds[i][0] for i in idxs]).to(device)  # (n, 10, 2)
    tgt_batch = torch.stack([val_ds[i][1] for i in idxs]).to(device)

    samples = ddim_sample(model, cond_batch, n_steps=ddim_steps, device=device)

    cond_d = full_ds.denorm(cond_batch).cpu().numpy()
    tgt_d = full_ds.denorm(tgt_batch).cpu().numpy()
    samp_d = full_ds.denorm(samples).cpu().numpy()

    # Evaluate the splines on dense u for plotting
    u = np.linspace(0, 1, 80)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, ci, ti, si in zip(axes.flat, cond_d, tgt_d, samp_d):
        cond_curve = eval_spline(ci, u)
        tgt_curve = eval_spline(ti, u)
        samp_curve = eval_spline(si, u)
        ax.plot(cond_curve[:, 0], cond_curve[:, 1], "b-", lw=2, label="cond [0,1.5s]")
        ax.plot(tgt_curve[:, 0], tgt_curve[:, 1], "g-", lw=2, label="GT future")
        ax.plot(samp_curve[:, 0], samp_curve[:, 1], "r--", lw=2, label="DDIM sample")
        ax.scatter(ci[:, 0], ci[:, 1], c="blue", s=10)
        ax.scatter(ti[:, 0], ti[:, 1], c="green", s=10)
        ax.scatter(si[:, 0], si[:, 1], c="red", s=10, marker="x")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"DDIM samples ({ddim_steps} steps) — cond/GT/pred in normalised CP space",
    )
    fig.tight_layout()
    fig.savefig("samples.png", dpi=120)
    print("Saved samples.png")

    # Quantitative: per-CP MSE on the denormalised samples
    err = ((samp_d - tgt_d) ** 2).mean()
    print(f"Sample MSE vs GT (denorm CP space): {err:.4f}")


if __name__ == "__main__":
    main()
