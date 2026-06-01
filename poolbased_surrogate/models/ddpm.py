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


class ResidualConvBlock1D(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode="circular",
            ),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualDDPMNet1D(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, kernel_size: int, n_blocks: int):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
            nn.SiLU(),
        )
        dilations = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.Sequential(
            *[
                ResidualConvBlock1D(hidden, kernel_size, dilations[i % len(dilations)])
                for i in range(n_blocks)
            ]
        )
        self.out_proj = nn.Conv1d(hidden, out_channels, kernel_size, padding=kernel_size // 2, padding_mode="circular")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.blocks(self.in_proj(x)))


def _build_conv_net(in_channels: int, hidden: int, kernel_size: int, n_residual_blocks: int) -> nn.Module:
    if n_residual_blocks > 0:
        return ResidualDDPMNet1D(in_channels, hidden, 1, kernel_size, n_residual_blocks)
    return nn.Sequential(
        nn.Conv1d(in_channels, hidden, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
        nn.SiLU(),
        nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
        nn.SiLU(),
        nn.Conv1d(hidden, 1, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
    )


class DDPM1D(nn.Module):
    """1D state diffusion model with optional loss/quantile conditioning and PDE-parameter handling.

    param_mode:
      - "none":      generate state only; params handled outside the model.
      - "condition": generate state conditioned on (clean) physical params + optional loss/quantile.
      - "generate":  jointly generate state and params via a separate MLP head.

    loss conditioning modes (mutually exclusive):
      - n_quantiles == 0: scalar float loss in [0, 1.5] (original behaviour).
      - n_quantiles > 0:  integer quantile label in {0, …, n_quantiles-1} routed through an
                          nn.Embedding; quant_embed_dim defaults to max(8, hidden//8).
    """

    def __init__(
        self,
        resolution: int,
        hidden: int = 64,
        steps: int = 64,
        loss_conditional: bool = True,
        param_dim: int = 0,
        param_mode: str = "none",
        residual_blocks: int = 0,
        kernel_size: int = 5,
        n_quantiles: int = 0,
        quant_embed_dim: int = 0,
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
        self.n_quantiles = int(n_quantiles)
        _qed = int(quant_embed_dim) if quant_embed_dim > 0 else max(8, hidden // 8)
        self.quant_embed_dim = _qed if self.n_quantiles > 0 else 0
        if self.n_quantiles > 0:
            self.quantile_embed = nn.Embedding(self.n_quantiles, self.quant_embed_dim)

        n_loss = self.quant_embed_dim if self.n_quantiles > 0 else (1 if self.loss_conditional else 0)
        n_pcond = self.param_dim if self.use_param_cond else 0
        self.cond = nn.Sequential(
            nn.Linear(self.temb_dim + n_loss + n_pcond, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.net = _build_conv_net(1 + hidden, hidden, kernel_size, residual_blocks)
        if self.generate_params:
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

    def _loss_features(self, loss: torch.Tensor | None, quantile_label: torch.Tensor | None) -> torch.Tensor | None:
        if self.n_quantiles > 0:
            if quantile_label is None:
                raise ValueError("n_quantiles > 0 requires quantile_label.")
            return self.quantile_embed(quantile_label.long())
        if self.loss_conditional:
            if loss is None:
                raise ValueError("Loss-conditional DDPM requires loss values.")
            return loss.view(loss.shape[0], 1)
        return None

    def _state_cond(
        self,
        t: torch.Tensor,
        loss: torch.Tensor | None,
        params: torch.Tensor | None,
        quantile_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [timestep_embedding(t, self.temb_dim)]
        lf = self._loss_features(loss, quantile_label)
        if lf is not None:
            parts.append(lf)
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
        quantile_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cond = self._state_cond(t, loss, params, quantile_label)[:, :, None].expand(-1, -1, noisy.shape[-1])
        state_eps = self.net(torch.cat([noisy, cond], dim=1))
        param_eps = None
        if self.generate_params:
            lf = self._loss_features(loss, quantile_label)
            pparts = [timestep_embedding(t, self.temb_dim)]
            if lf is not None:
                pparts.append(lf)
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
        quantile_label: torch.Tensor | None = None,
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
        pred, pred_p = self(noisy, t, loss, state_in, quantile_label)
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
        temperature: float = 1.0,
        quantile_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = device or self.beta.device
        temperature = float(temperature)
        x = temperature * torch.randn(n, 1, self.resolution, device=device)
        if loss is not None:
            loss = loss.to(device).view(n, 1)
        if quantile_label is not None:
            quantile_label = quantile_label.to(device).long()
        if self.use_param_cond and params is not None:
            params = params.to(device).view(n, self.param_dim)
        p = temperature * torch.randn(n, self.param_dim, device=device) if self.generate_params else None
        for step in reversed(range(self.steps)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            state_in = p if self.generate_params else params
            eps, eps_p = self(x, t, loss, state_in, quantile_label)
            alpha = self.alpha[step]
            ab = self.alpha_bar[step]
            beta = self.beta[step]
            x = (x - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
            if self.generate_params:
                p = (p - beta / torch.sqrt(1.0 - ab) * eps_p) / torch.sqrt(alpha)
            if step > 0:
                noise = temperature * torch.sqrt(beta) * torch.randn_like(x)
                x = x + noise
                if self.generate_params:
                    p = p + temperature * torch.sqrt(beta) * torch.randn_like(p)
        x = x.clamp(-6.0, 6.0)
        if self.generate_params:
            p = p.clamp(-1.5, 1.5)
        return x, p


class FlowMatching1D(nn.Module):
    """Conditional 1D flow matching generator.

    Shares the same public API as DDPM1D.  Loss conditioning modes:
      - n_quantiles == 0: scalar float loss in [0, 1.5].
      - n_quantiles > 0:  integer quantile label routed through nn.Embedding.
    """

    def __init__(
        self,
        resolution: int,
        hidden: int = 64,
        steps: int = 16,
        loss_conditional: bool = True,
        param_dim: int = 0,
        param_mode: str = "none",
        residual_blocks: int = 0,
        kernel_size: int = 5,
        n_quantiles: int = 0,
        quant_embed_dim: int = 0,
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
        self.n_quantiles = int(n_quantiles)
        _qed = int(quant_embed_dim) if quant_embed_dim > 0 else max(8, hidden // 8)
        self.quant_embed_dim = _qed if self.n_quantiles > 0 else 0
        if self.n_quantiles > 0:
            self.quantile_embed = nn.Embedding(self.n_quantiles, self.quant_embed_dim)

        n_loss = self.quant_embed_dim if self.n_quantiles > 0 else (1 if self.loss_conditional else 0)
        n_pcond = self.param_dim if self.use_param_cond else 0
        self.cond = nn.Sequential(
            nn.Linear(self.temb_dim + n_loss + n_pcond, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.net = _build_conv_net(1 + hidden, hidden, kernel_size, residual_blocks)
        if self.generate_params:
            self.param_cond = nn.Sequential(
                nn.Linear(self.temb_dim + n_loss + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.param_out = nn.Linear(hidden, self.param_dim)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t.float() * max(self.steps - 1, 1), self.temb_dim)

    def _loss_features(self, loss: torch.Tensor | None, quantile_label: torch.Tensor | None) -> torch.Tensor | None:
        if self.n_quantiles > 0:
            if quantile_label is None:
                raise ValueError("n_quantiles > 0 requires quantile_label.")
            return self.quantile_embed(quantile_label.long())
        if self.loss_conditional:
            if loss is None:
                raise ValueError("Loss-conditional flow matching requires loss values.")
            return loss.view(loss.shape[0], 1)
        return None

    def _state_cond(
        self,
        t: torch.Tensor,
        loss: torch.Tensor | None,
        params: torch.Tensor | None,
        quantile_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self._time_embedding(t)]
        lf = self._loss_features(loss, quantile_label)
        if lf is not None:
            parts.append(lf)
        if self.use_param_cond:
            if params is None:
                raise ValueError("param_mode requires param values.")
            parts.append(params.view(params.shape[0], self.param_dim))
        return self.cond(torch.cat(parts, dim=1))

    def forward(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
        loss: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        quantile_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cond = self._state_cond(t, loss, params, quantile_label)[:, :, None].expand(-1, -1, state.shape[-1])
        state_v = self.net(torch.cat([state, cond], dim=1))
        param_v = None
        if self.generate_params:
            lf = self._loss_features(loss, quantile_label)
            pparts = [self._time_embedding(t)]
            if lf is not None:
                pparts.append(lf)
            pparts.append(state.mean(dim=2))
            param_v = self.param_out(self.param_cond(torch.cat(pparts, dim=1)))
        return state_v, param_v

    def training_loss(
        self,
        clean: torch.Tensor,
        loss: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        param_loss_weight: float = 1.0,
        quantile_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = clean.shape[0]
        t = torch.rand((bsz,), device=clean.device)
        t_state = t.view(bsz, 1, 1)
        noise = torch.randn_like(clean)
        mixed = (1.0 - t_state) * noise + t_state * clean
        target_v = clean - noise
        if self.generate_params:
            t_param = t.view(bsz, 1)
            param_noise = torch.randn_like(params)
            mixed_params = (1.0 - t_param) * param_noise + t_param * params
            state_in = mixed_params
        else:
            state_in = params if self.use_param_cond else None
        pred_v, pred_param_v = self(mixed, t, loss, state_in, quantile_label)
        per_sample = torch.mean((pred_v - target_v) ** 2, dim=(1, 2))
        if self.generate_params:
            target_param_v = params - param_noise
            per_sample = per_sample + param_loss_weight * torch.mean((pred_param_v - target_param_v) ** 2, dim=1)
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
        temperature: float = 1.0,
        quantile_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = device or next(self.parameters()).device
        temperature = float(temperature)
        x = temperature * torch.randn(n, 1, self.resolution, device=device)
        if loss is not None:
            loss = loss.to(device).view(n, 1)
        if quantile_label is not None:
            quantile_label = quantile_label.to(device).long()
        if self.use_param_cond and params is not None:
            params = params.to(device).view(n, self.param_dim)
        p = temperature * torch.randn(n, self.param_dim, device=device) if self.generate_params else None
        dt = 1.0 / max(self.steps, 1)
        for step in range(self.steps):
            t = torch.full((n,), step / max(self.steps - 1, 1), device=device, dtype=torch.float32)
            state_in = p if self.generate_params else params
            v, pv = self(x, t, loss, state_in, quantile_label)
            x = x + dt * v
            if self.generate_params:
                p = p + dt * pv
        x = x.clamp(-6.0, 6.0)
        if self.generate_params:
            p = p.clamp(-1.5, 1.5)
        return x, p
