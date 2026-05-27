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
    """1D state diffusion model with optional loss conditioning and PDE-parameter handling.

    param_mode:
      - "none":      generate state only; params handled outside the model.
      - "condition": generate state conditioned on (clean) physical params + optional loss.
      - "generate":  jointly generate state and params; a separate MLP head denoises the
                     low-dimensional param vector, coupled to the state via shared conditioning.
    """

    def __init__(
        self,
        resolution: int,
        hidden: int = 64,
        steps: int = 64,
        loss_conditional: bool = True,
        param_dim: int = 0,
        param_mode: str = "none",
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.steps = int(steps)
        self.temb_dim = int(hidden)
        self.loss_conditional = bool(loss_conditional)
        self.param_dim = int(param_dim)
        self.param_mode = str(param_mode)
        self.generate_params = self.param_mode == "generate"
        self.use_param_cond = self.param_mode in ("condition", "generate") and self.param_dim > 0

        n_loss = 1 if self.loss_conditional else 0
        n_pcond = self.param_dim if self.use_param_cond else 0
        self.cond = nn.Sequential(
            nn.Linear(self.temb_dim + n_loss + n_pcond, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.net = nn.Sequential(
            nn.Conv1d(1 + hidden, hidden, 5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, 5, padding=2, padding_mode="circular"),
        )
        if self.generate_params:
            # The param head sees the timestep, optional loss, and a pooled summary of the
            # (noisy) state so the generated params stay coupled to the generated field.
            self.param_cond = nn.Sequential(
                nn.Linear(self.temb_dim + n_loss + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.param_out = nn.Linear(hidden, self.param_dim)

        beta = torch.linspace(1e-4, 0.02, self.steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def _state_cond(self, t: torch.Tensor, loss: torch.Tensor | None, params: torch.Tensor | None) -> torch.Tensor:
        parts = [timestep_embedding(t, self.temb_dim)]
        if self.loss_conditional:
            if loss is None:
                raise ValueError("Loss-conditional DDPM requires loss values.")
            parts.append(loss.view(loss.shape[0], 1))
        if self.use_param_cond:
            if params is None:
                raise ValueError("param_mode requires param values.")
            parts.append(params.view(params.shape[0], self.param_dim))
        return self.cond(torch.cat(parts, dim=1))

    def forward(
        self,
        noisy: torch.Tensor,
        t: torch.Tensor,
        loss: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cond = self._state_cond(t, loss, params)[:, :, None].expand(-1, -1, noisy.shape[-1])
        state_eps = self.net(torch.cat([noisy, cond], dim=1))
        param_eps = None
        if self.generate_params:
            pparts = [timestep_embedding(t, self.temb_dim)]
            if self.loss_conditional:
                pparts.append(loss.view(loss.shape[0], 1))
            pparts.append(noisy.mean(dim=2))
            param_eps = self.param_out(self.param_cond(torch.cat(pparts, dim=1)))
        return state_eps, param_eps

    def training_loss(
        self,
        clean: torch.Tensor,
        loss: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        param_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        bsz = clean.shape[0]
        t = torch.randint(0, self.steps, (bsz,), device=clean.device)
        ab = self.alpha_bar[t].view(bsz, 1, 1)
        eps = torch.randn_like(clean)
        noisy = ab.sqrt() * clean + (1.0 - ab).sqrt() * eps
        if self.generate_params:
            ab_p = self.alpha_bar[t].view(bsz, 1)
            eps_p = torch.randn_like(params)
            noisy_params = ab_p.sqrt() * params + (1.0 - ab_p).sqrt() * eps_p
            state_in = noisy_params
        else:
            state_in = params if self.use_param_cond else None
        pred, pred_p = self(noisy, t, loss, state_in)
        per_sample = torch.mean((pred - eps) ** 2, dim=(1, 2))
        if self.generate_params:
            per_sample = per_sample + param_loss_weight * torch.mean((pred_p - eps_p) ** 2, dim=1)
        if sample_weight is not None:
            w = sample_weight.view(-1).to(per_sample.device)
            return torch.sum(per_sample * w) / (torch.sum(w) + 1e-8)
        return torch.mean(per_sample)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        loss: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = device or self.beta.device
        x = torch.randn(n, 1, self.resolution, device=device)
        if loss is not None:
            loss = loss.to(device).view(n, 1)
        if self.use_param_cond and params is not None:
            params = params.to(device).view(n, self.param_dim)
        p = torch.randn(n, self.param_dim, device=device) if self.generate_params else None
        for step in reversed(range(self.steps)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            state_in = p if self.generate_params else params
            eps, eps_p = self(x, t, loss, state_in)
            alpha = self.alpha[step]
            ab = self.alpha_bar[step]
            beta = self.beta[step]
            x = (x - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
            if self.generate_params:
                p = (p - beta / torch.sqrt(1.0 - ab) * eps_p) / torch.sqrt(alpha)
            if step > 0:
                noise = torch.sqrt(beta) * torch.randn_like(x)
                x = x + noise
                if self.generate_params:
                    p = p + torch.sqrt(beta) * torch.randn_like(p)
        x = x.clamp(-6.0, 6.0)
        if self.generate_params:
            p = p.clamp(-1.5, 1.5)
        return x, p
