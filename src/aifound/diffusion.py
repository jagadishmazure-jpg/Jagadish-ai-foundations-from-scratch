"""A toy diffusion model (PyTorch, CPU) on 2-D points.

Target "images" are points on a ring of radius 1. Forward process: blend a point with Gaussian noise
over T steps until it is pure noise. A small network learns to predict the noise that was added at
each step. Generation runs the other way: start from pure noise and remove a little predicted noise
per step (DDPM sampling). We measure how far samples sit from the ring along the way.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .pii import SafeError

T = 50


def ring(n: int, g: torch.Generator) -> torch.Tensor:
    a = torch.rand(n, generator=g) * 2 * math.pi
    return torch.stack([a.cos(), a.sin()], 1) + 0.02 * torch.randn(n, 2, generator=g)


def schedule(t_steps: int = T):
    beta = torch.linspace(1e-4, 0.2, t_steps)
    alpha = 1 - beta
    return beta, alpha, torch.cumprod(alpha, 0)


class Denoiser(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, (t.float() / T).unsqueeze(1)], 1))


def ring_error(x: torch.Tensor) -> float:
    """Mean |distance to origin - 1|: 0 means every point is on the ring."""
    return float((x.norm(dim=1) - 1).abs().mean())


def train(steps: int = 3000, seed: int = 0) -> Denoiser:
    if steps < 1:
        raise SafeError("steps must be positive")
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = Denoiser()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    _, _, abar = schedule()
    for _ in range(steps):
        x0 = ring(256, g)
        t = torch.randint(0, T, (256,), generator=g)
        eps = torch.randn(256, 2, generator=g)
        xt = abar[t].sqrt().unsqueeze(1) * x0 + (1 - abar[t]).sqrt().unsqueeze(1) * eps
        loss = nn.functional.mse_loss(model(xt, t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def sample(model: Denoiser, n: int = 1000, seed: int = 1) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    beta, alpha, abar = schedule()
    x = torch.randn(n, 2, generator=g)
    frames = [x.clone()]
    for t in reversed(range(T)):
        eps = model(x, torch.full((n,), t))
        x = (x - beta[t] / (1 - abar[t]).sqrt() * eps) / alpha[t].sqrt()
        if t > 0:
            x = x + beta[t].sqrt() * torch.randn(n, 2, generator=g)
        frames.append(x.clone())
    return frames


def run(seed: int = 0, steps: int = 3000) -> dict:
    model = train(steps, seed)
    frames = sample(model)
    picks = [0, 10, 25, 40, 50]
    g = torch.Generator().manual_seed(9)
    return {
        "train_steps": steps,
        "denoise_steps": T,
        "samples": 1000,
        "ring_error_by_step": {str(i): round(ring_error(frames[i]), 3) for i in picks},
        "ring_error_real_data": round(ring_error(ring(1000, g)), 3),
        "_frames": [frames[i] for i in picks],
    }
