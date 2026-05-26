from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class DDPM1D(nn.Module):
    def __init__(
        self,
        resolution: int,
        hidden: int = 64,
        steps: int = 64,
        conditional: bool = True,
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.steps = int(steps)
        self.conditional = bool(conditional)
        cond_in = 1 + hidden if self.conditional else hidden
        self.cond = nn.Sequential(nn.Linear(cond_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Conv1d(1 + hidden, hidden, 5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, 5, padding=2, padding_mode="circular"),
        )
        beta = torch.linspace(1e-4, 0.02, self.steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(
        self,
        noisy: torch.Tensor,
        t: torch.Tensor,
        loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb_dim = self.cond[0].in_features - (1 if self.conditional else 0)
        emb = timestep_embedding(t, emb_dim)
        if self.conditional:
            if loss is None:
                raise ValueError("Conditional DDPM requires loss values.")
            cond_input = torch.cat([loss, emb], dim=1)
        else:
            cond_input = emb
        cond = self.cond(cond_input)[:, :, None].expand(-1, -1, noisy.shape[-1])
        return self.net(torch.cat([noisy, cond], dim=1))

    def training_loss(
        self,
        clean: torch.Tensor,
        loss: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = clean.shape[0]
        t = torch.randint(0, self.steps, (bsz,), device=clean.device)
        eps = torch.randn_like(clean)
        ab = self.alpha_bar[t].view(bsz, 1, 1)
        noisy = ab.sqrt() * clean + (1.0 - ab).sqrt() * eps
        pred = self(noisy, t, loss)
        per_sample = torch.mean((pred - eps) ** 2, dim=(1, 2))
        if sample_weight is not None:
            w = sample_weight.view(-1).to(per_sample.device)
            return torch.sum(per_sample * w) / (torch.sum(w) + 1e-8)
        return torch.mean(per_sample)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        losses: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        x = torch.randn(n, 1, self.resolution, device=device)
        if losses is not None:
            losses = losses.to(device).view(n, 1)
        for step in reversed(range(self.steps)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            eps = self(x, t, losses)
            alpha = self.alpha[step]
            ab = self.alpha_bar[step]
            beta = self.beta[step]
            x = (x - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x.clamp(-5.0, 5.0)
