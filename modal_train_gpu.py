import modal
import time
import pickle
import io
import numpy as np
import torch

app = modal.App("test-train-standalone")
train_image = modal.Image.debian_slim(python_version="3.13").pip_install("torch", "numpy", "scipy")

@app.function(image=train_image, gpu="T4", timeout=1800)
def train_gpu(model_py_bytes, checkpoint_bytes, frame_data, epochs, lr, batch_size, T, hidden, n_blocks, print_interval=1):
    import sys, os
    os.makedirs("/root/train", exist_ok=True)
    with open("/root/train/model.py", "wb") as f:
        f.write(model_py_bytes)
    sys.path.insert(0, "/root/train")
    os.chdir("/root/train")
    
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from model import DiffusionMLP, CosineSchedule, COND_DIM, STATE_DIM, CFG_START, CFG_END
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, hidden={hidden}")
    
    with open("/tmp/ckpt.pt", "wb") as f:
        f.write(checkpoint_bytes)
    ckpt = torch.load("/tmp/ckpt.pt", weights_only=False, map_location=device)
    
    model = DiffusionMLP(hidden=hidden, n_blocks=n_blocks).to(device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        except Exception:
            pass
    
    frame_list = pickle.loads(frame_data)
    conds = np.array([f[0] for f in frame_list], dtype=np.float32)
    targets = np.array([f[1] for f in frame_list], dtype=np.float32)
    print(f"Training on {len(conds)} frames for {epochs} epochs")
    
    x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32).to(device)
    x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32).to(device)
    
    schedule = CosineSchedule(T=T)
    alpha_bar_all = torch.tensor(schedule.alpha_bar, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    class _DS(Dataset):
        def __init__(s, c, t):
            s.c, s.t = c, t
        def __len__(s):
            return len(s.c)
        def __getitem__(s, i):
            c = s.c[i].copy()
            # No CFG dropout — outcome is a regular conditioning input
            return torch.tensor(c, dtype=torch.float32), torch.tensor(s.t[i], dtype=torch.float32)
    
    loader = DataLoader(_DS(conds, targets), batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for cb, tb in loader:
            cb, tb = cb.to(device), tb.to(device)
            x0 = (tb - x_mean) / x_std
            t = torch.randint(1, T+1, (cb.shape[0],), device=device)
            ab = alpha_bar_all[t].unsqueeze(1)
            noise = torch.randn_like(x0)
            x_noisy = torch.sqrt(ab)*x0 + torch.sqrt(1-ab)*noise
            loss = nn.functional.mse_loss(model(x_noisy, cb, t), noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch+1) % print_interval == 0:
            print(f"  epoch {epoch+1}/{epochs} loss={loss.item():.4f}")
    
    model.cpu()
    buf = io.BytesIO()
    torch.save({"model_state_dict": model.state_dict(), "x_mean": x_mean.cpu().numpy(), "x_std": x_std.cpu().numpy(), "T": T, "epochs": epochs, "hidden": hidden, "n_blocks": n_blocks}, buf)
    return buf.getvalue()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", required=True, help="Path to pickled frame data")
    parser.add_argument("--output", required=True, help="Path to save new checkpoint")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--T", type=int, default=100)
    args = parser.parse_args()

    with open("model.py", "rb") as f:
        mpy = f.read()
    with open(args.checkpoint, "rb") as f:
        ckpt = f.read()
    with open(args.frames, "rb") as f:
        frames = f.read()  # already pickled

    print(f"ckpt={len(ckpt)/1e6:.1f}MB frames={len(frames)/1e3:.0f}KB", flush=True)
    t0 = time.time()
    with app.run():
        r = train_gpu.remote(
            mpy, ckpt, frames,
            args.epochs, args.lr, args.batch_size, args.T,
            args.hidden, args.n_blocks,
            max(50, args.epochs // 10),
        )

    with open(args.output, "wb") as f:
        f.write(r)
    print(f"Done in {time.time()-t0:.1f}s, saved {len(r)/1e6:.1f}MB to {args.output}")
