"""DDPM training noise + DDIM deterministic sampler.

We use a standard linear beta schedule and predict noise (eps).
The DDIM update is the deterministic eta=0 case from Song et al. 2021.
"""

import torch

T_TRAIN = 1000


def make_schedule(T: int = T_TRAIN, device: torch.device = torch.device("cpu")):
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def q_sample(x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor):
    """Forward noising: x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) noise."""
    ab = alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
    noise = torch.randn_like(x0)
    x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
    return x_t, noise


@torch.no_grad()
def ddim_sample(
    model,
    cond_cps: torch.Tensor,   # (B, 10, 2)
    n_steps: int = 50,
    T: int = T_TRAIN,
    device: torch.device = torch.device("cpu"),
):
    """Deterministic DDIM sampler with eta=0."""
    _, _, alpha_bar = make_schedule(T, device=device)
    B = cond_cps.shape[0]
    x = torch.randn(B, 10, 2, device=device)

    # Sub-sequence of timesteps from T-1 down to 0
    step_indices = torch.linspace(T - 1, 0, n_steps + 1, device=device).long()

    for i in range(n_steps):
        t_cur = step_indices[i]
        t_next = step_indices[i + 1]
        ab_cur = alpha_bar[t_cur]
        ab_next = alpha_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

        t_batch = torch.full((B,), int(t_cur), device=device, dtype=torch.long)
        eps = model(cond_cps, x, t_batch)

        # Predict x0 from eps
        x0_pred = (x - (1.0 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        # Deterministic step (eta=0)
        x = ab_next.sqrt() * x0_pred + (1.0 - ab_next).sqrt() * eps

    return x
