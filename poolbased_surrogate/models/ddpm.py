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
    """Dilated residual conv block with optional FiLM conditioning.

    With film_cond_dim > 0 a zero-initialised Linear maps the conditioning vector
    to per-channel (gamma, beta) applied to the activations of the first conv as
    h * (1 + gamma) + beta (identity at init). The `net` Sequential keeps the
    exact same parameter keys as the unconditioned block, so checkpoints made
    without FiLM stay loadable.
    """

    def __init__(self, hidden: int, kernel_size: int, dilation: int, film_cond_dim: int = 0):
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
        self.film = None
        if film_cond_dim > 0:
            self.film = nn.Linear(film_cond_dim, 2 * hidden)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if self.film is None or cond is None:
            return x + self.net(x)
        h = self.net[0](x)
        gamma, beta = self.film(cond)[:, :, None].chunk(2, dim=1)
        h = h * (1.0 + gamma) + beta
        for layer in self.net[1:]:
            h = layer(h)
        return x + h


class ResidualDDPMNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        kernel_size: int,
        n_blocks: int,
        film_cond_dim: int = 0,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
            nn.SiLU(),
        )
        dilations = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.Sequential(
            *[
                ResidualConvBlock1D(hidden, kernel_size, dilations[i % len(dilations)], film_cond_dim)
                for i in range(n_blocks)
            ]
        )
        self.out_proj = nn.Conv1d(hidden, out_channels, kernel_size, padding=kernel_size // 2, padding_mode="circular")

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(h)


def _build_conv_net(
    in_channels: int, hidden: int, kernel_size: int, n_residual_blocks: int, film_cond_dim: int = 0
) -> nn.Module:
    if n_residual_blocks > 0:
        return ResidualDDPMNet1D(in_channels, hidden, 1, kernel_size, n_residual_blocks, film_cond_dim)
    if film_cond_dim > 0:
        raise ValueError("cond_mode='film' requires residual_blocks > 0 (FiLM is per-residual-block)")
    return nn.Sequential(
        nn.Conv1d(in_channels, hidden, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
        nn.SiLU(),
        nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
        nn.SiLU(),
        nn.Conv1d(hidden, 1, kernel_size, padding=kernel_size // 2, padding_mode="circular"),
    )


class GeneratorStateMixin:
    """Persistent conditioning state shared by the generators.

    Registers buffers (so they survive checkpoint save/load) for:
      - quantile bin edges of the difficulty signal,
      - the EMA/frozen q05/q95 normalization stats of the difficulty signal,
      - the sqrt power-spectral-density profile of the coloured-noise base prior.
    """

    def _init_generator_state(self, resolution: int, n_quantiles: int) -> None:
        self.register_buffer("quantile_edges_buf", torch.zeros(max(n_quantiles + 1, 1)))
        self.register_buffer("difficulty_lo", torch.tensor(0.0))
        self.register_buffer("difficulty_hi", torch.tensor(0.0))
        self.register_buffer("difficulty_stats_ready", torch.tensor(0.0))
        self.register_buffer("noise_psd_sqrt", torch.zeros(resolution // 2 + 1))
        self.register_buffer("noise_psd_ready", torch.tensor(0.0))
        self.quantile_edges = None  # numpy mirror consumed by generator_eval

    def set_quantile_edges(self, edges) -> None:
        import numpy as _np

        edges = _np.asarray(edges, dtype=_np.float32).reshape(-1)
        self.quantile_edges = edges
        if edges.shape[0] == self.quantile_edges_buf.shape[0]:
            self.quantile_edges_buf.copy_(torch.from_numpy(edges).to(self.quantile_edges_buf.device))

    def restore_generator_state(self) -> None:
        """Re-populate the numpy mirrors after a checkpoint load."""
        if float(self.quantile_edges_buf.abs().sum()) > 0:
            self.quantile_edges = self.quantile_edges_buf.detach().cpu().numpy()

    def update_difficulty_stats(self, lo: float, hi: float, mode: str = "ema", momentum: float = 0.7) -> tuple[float, float]:
        """Track the q05/q95 normalization stats across rounds; returns the stats to use."""
        if mode == "per_round" or float(self.difficulty_stats_ready) == 0.0:
            self.difficulty_lo.fill_(float(lo))
            self.difficulty_hi.fill_(float(hi))
            self.difficulty_stats_ready.fill_(1.0)
        elif mode == "ema":
            m = float(momentum)
            self.difficulty_lo.mul_(m).add_((1.0 - m) * float(lo))
            self.difficulty_hi.mul_(m).add_((1.0 - m) * float(hi))
        # mode == "frozen": keep the stored stats untouched
        return float(self.difficulty_lo), float(self.difficulty_hi)

    def set_noise_psd(self, ref_states: torch.Tensor, momentum: float = 0.7) -> None:
        """Fit the coloured-noise prior to reference (z-normalised) states [N, 1, L]."""
        spec = torch.fft.rfft(ref_states.float().squeeze(1), dim=-1)
        psd_sqrt = (spec.real**2 + spec.imag**2).mean(dim=0).clamp_min(1e-12).sqrt()
        psd_sqrt = psd_sqrt.to(self.noise_psd_sqrt.device)
        if float(self.noise_psd_ready) > 0:
            self.noise_psd_sqrt.mul_(momentum).add_((1.0 - momentum) * psd_sqrt)
        else:
            self.noise_psd_sqrt.copy_(psd_sqrt)
            self.noise_psd_ready.fill_(1.0)

    def _base_noise(self, n: int, device, temperature: float = 1.0) -> torch.Tensor:
        """Draw base noise: white N(0,I), or unit-variance coloured Gaussian when a PSD is set."""
        white = torch.randn(n, 1, self.resolution, device=device)
        if float(self.noise_psd_ready) == 0.0:
            return temperature * white
        spec = torch.fft.rfft(white.squeeze(1), dim=-1) * self.noise_psd_sqrt.to(device)
        x = torch.fft.irfft(spec, n=self.resolution, dim=-1).unsqueeze(1)
        x = x / (x.std(dim=(1, 2), keepdim=True) + 1e-8)
        return temperature * x

    def _cfg_drop_mask(self, bsz: int, device, quantile_label) -> torch.Tensor | None:
        """Training-time CFG label dropout mask (None when CFG is inactive)."""
        if (
            getattr(self, "null_quant_embed", None) is None
            or float(getattr(self, "cfg_dropout", 0.0)) <= 0.0
            or quantile_label is None
            or not self.training
        ):
            return None
        return torch.rand(bsz, device=device) < float(self.cfg_dropout)

    def _cfg_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        loss: torch.Tensor | None,
        params: torch.Tensor | None,
        quantile_label: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Denoiser output with classifier-free guidance on the quantile label.

        With cfg_scale == 1.0 (or no label / no null embedding) this is exactly one
        conditional forward pass — bit-identical to the unguided model. Otherwise a
        second pass with the NULL label embedding is combined per CFG:
        out = null + cfg_scale * (cond - null). Guidance acts only on the predicted
        velocity/eps (state and, when present, param heads); the sampler's init
        logic (coloured-noise prior, edit-mode anchors) is unchanged.
        """
        cond_out, cond_p = self(x, t, loss, params, quantile_label)
        scale = float(getattr(self, "cfg_scale", 1.0))
        if scale == 1.0 or quantile_label is None or getattr(self, "null_quant_embed", None) is None:
            return cond_out, cond_p
        drop_all = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        null_out, null_p = self(x, t, loss, params, quantile_label, cond_drop_mask=drop_all)
        out = null_out + scale * (cond_out - null_out)
        p = None if cond_p is None else null_p + scale * (cond_p - null_p)
        return out, p


class DDPM1D(GeneratorStateMixin, nn.Module):
    """1D state diffusion model with optional loss/quantile conditioning and PDE-parameter handling.

    param_mode:
      - "none":      generate state only; params handled outside the model.
      - "condition": generate state conditioned on (clean) physical params + optional loss/quantile.
      - "generate":  jointly generate state and params via a separate MLP head.

    loss conditioning modes (mutually exclusive):
      - n_quantiles == 0: scalar float loss in [0, 1.5] (original behaviour).
      - n_quantiles > 0:  integer quantile label in {0, …, n_quantiles-1} routed through an
                          nn.Embedding; quant_embed_dim defaults to max(8, hidden//8).

    Conditioning strength (both default-off, see DDPMConfig):
      - cfg_dropout/cfg_scale: classifier-free guidance on the quantile label. A
        separate learned null_quant_embed parameter is created ONLY when CFG is
        enabled (cfg_dropout > 0 or cfg_scale != 1.0) so default state_dicts are
        unchanged; checkpoint loading tolerates its absence (checkpoint.py).
      - cond_mode: "concat" (legacy input-channel concat) or "film" (keeps the
        concat AND adds per-residual-block FiLM scale-shift on the cond vector).
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
        cfg_dropout: float = 0.0,
        cfg_scale: float = 1.0,
        cond_mode: str = "concat",
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
        self.cfg_dropout = float(cfg_dropout)
        self.cfg_scale = float(cfg_scale)
        self.cond_mode = str(cond_mode)
        if self.cond_mode not in ("concat", "film"):
            raise ValueError(f"Unknown cond_mode {cond_mode!r}; expected 'concat' or 'film'.")
        self.null_quant_embed = None
        if self.n_quantiles > 0 and (self.cfg_dropout > 0.0 or self.cfg_scale != 1.0):
            self.null_quant_embed = nn.Parameter(torch.zeros(self.quant_embed_dim))

        n_loss = self.quant_embed_dim if self.n_quantiles > 0 else (1 if self.loss_conditional else 0)
        n_pcond = self.param_dim if self.use_param_cond else 0
        self.cond = nn.Sequential(
            nn.Linear(self.temb_dim + n_loss + n_pcond, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.net = _build_conv_net(
            1 + hidden, hidden, kernel_size, residual_blocks,
            film_cond_dim=hidden if self.cond_mode == "film" else 0,
        )
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
        self._init_generator_state(self.resolution, self.n_quantiles)

    def _loss_features(
        self,
        loss: torch.Tensor | None,
        quantile_label: torch.Tensor | None,
        cond_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.n_quantiles > 0:
            if quantile_label is None:
                raise ValueError("n_quantiles > 0 requires quantile_label.")
            emb = self.quantile_embed(quantile_label.long())
            if cond_drop_mask is not None:
                if self.null_quant_embed is None:
                    raise ValueError("CFG label dropout requires cfg_dropout > 0 or cfg_scale != 1.0.")
                emb = torch.where(cond_drop_mask.view(-1, 1), self.null_quant_embed.view(1, -1), emb)
            return emb
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
        cond_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [timestep_embedding(t, self.temb_dim)]
        lf = self._loss_features(loss, quantile_label, cond_drop_mask)
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
        cond_drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cond_vec = self._state_cond(t, loss, params, quantile_label, cond_drop_mask)
        cond = cond_vec[:, :, None].expand(-1, -1, noisy.shape[-1])
        x_in = torch.cat([noisy, cond], dim=1)
        state_eps = self.net(x_in, cond_vec) if self.cond_mode == "film" else self.net(x_in)
        param_eps = None
        if self.generate_params:
            lf = self._loss_features(loss, quantile_label, cond_drop_mask)
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
        drop_mask = self._cfg_drop_mask(bsz, clean.device, quantile_label)
        pred, pred_p = self(noisy, t, loss, state_in, quantile_label, drop_mask)
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
        init_state: torch.Tensor | None = None,
        t_start: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if init_state is not None and t_start > 0.0:
            raise NotImplementedError("edit-mode sampling is only implemented for FlowMatching1D")
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
            eps, eps_p = self._cfg_output(x, t, loss, state_in, quantile_label)
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


class FlowMatching1D(GeneratorStateMixin, nn.Module):
    """Conditional 1D flow matching generator.

    Shares the same public API as DDPM1D.  Loss conditioning modes:
      - n_quantiles == 0: scalar float loss in [0, 1.5].
      - n_quantiles > 0:  integer quantile label routed through nn.Embedding.

    Distribution priors (GeneratorStateMixin): when a noise PSD has been set via
    set_noise_psd, both training and sampling use a coloured Gaussian base whose
    spectrum matches the physical states, removing the white-noise high-frequency
    mismatch. sample() additionally supports SDEdit-style "edit" generation from
    real anchor states (init_state + t_start).

    Conditioning strength (both default-off, see DDPMConfig):
      - cfg_dropout/cfg_scale: classifier-free guidance on the quantile label. A
        separate learned null_quant_embed parameter is created ONLY when CFG is
        enabled (cfg_dropout > 0 or cfg_scale != 1.0) so default state_dicts are
        unchanged; checkpoint loading tolerates its absence (checkpoint.py).
      - cond_mode: "concat" (legacy input-channel concat) or "film" (keeps the
        concat AND adds per-residual-block FiLM scale-shift on the cond vector).
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
        cfg_dropout: float = 0.0,
        cfg_scale: float = 1.0,
        cond_mode: str = "concat",
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
        self.cfg_dropout = float(cfg_dropout)
        self.cfg_scale = float(cfg_scale)
        self.cond_mode = str(cond_mode)
        if self.cond_mode not in ("concat", "film"):
            raise ValueError(f"Unknown cond_mode {cond_mode!r}; expected 'concat' or 'film'.")
        self.null_quant_embed = None
        if self.n_quantiles > 0 and (self.cfg_dropout > 0.0 or self.cfg_scale != 1.0):
            self.null_quant_embed = nn.Parameter(torch.zeros(self.quant_embed_dim))

        n_loss = self.quant_embed_dim if self.n_quantiles > 0 else (1 if self.loss_conditional else 0)
        n_pcond = self.param_dim if self.use_param_cond else 0
        self.cond = nn.Sequential(
            nn.Linear(self.temb_dim + n_loss + n_pcond, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.net = _build_conv_net(
            1 + hidden, hidden, kernel_size, residual_blocks,
            film_cond_dim=hidden if self.cond_mode == "film" else 0,
        )
        if self.generate_params:
            self.param_cond = nn.Sequential(
                nn.Linear(self.temb_dim + n_loss + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.param_out = nn.Linear(hidden, self.param_dim)
        self._init_generator_state(self.resolution, self.n_quantiles)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t.float() * max(self.steps - 1, 1), self.temb_dim)

    def _loss_features(
        self,
        loss: torch.Tensor | None,
        quantile_label: torch.Tensor | None,
        cond_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.n_quantiles > 0:
            if quantile_label is None:
                raise ValueError("n_quantiles > 0 requires quantile_label.")
            emb = self.quantile_embed(quantile_label.long())
            if cond_drop_mask is not None:
                if self.null_quant_embed is None:
                    raise ValueError("CFG label dropout requires cfg_dropout > 0 or cfg_scale != 1.0.")
                emb = torch.where(cond_drop_mask.view(-1, 1), self.null_quant_embed.view(1, -1), emb)
            return emb
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
        cond_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self._time_embedding(t)]
        lf = self._loss_features(loss, quantile_label, cond_drop_mask)
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
        cond_drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cond_vec = self._state_cond(t, loss, params, quantile_label, cond_drop_mask)
        cond = cond_vec[:, :, None].expand(-1, -1, state.shape[-1])
        x_in = torch.cat([state, cond], dim=1)
        state_v = self.net(x_in, cond_vec) if self.cond_mode == "film" else self.net(x_in)
        param_v = None
        if self.generate_params:
            lf = self._loss_features(loss, quantile_label, cond_drop_mask)
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
        # Base distribution matches sampling: coloured (physical-spectrum) noise when set.
        noise = self._base_noise(bsz, clean.device)
        mixed = (1.0 - t_state) * noise + t_state * clean
        target_v = clean - noise
        if self.generate_params:
            t_param = t.view(bsz, 1)
            param_noise = torch.randn_like(params)
            mixed_params = (1.0 - t_param) * param_noise + t_param * params
            state_in = mixed_params
        else:
            state_in = params if self.use_param_cond else None
        drop_mask = self._cfg_drop_mask(bsz, clean.device, quantile_label)
        pred_v, pred_param_v = self(mixed, t, loss, state_in, quantile_label, drop_mask)
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
        init_state: torch.Tensor | None = None,
        t_start: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Integrate the learned velocity field from base noise (or, in edit mode,
        from a partially re-noised real anchor) to the data distribution.

        Edit mode (init_state + t_start in (0, 1)): x(t0) = (1-t0)*noise + t0*anchor,
        integrated over [t0, 1]. The anchor must already be z-normalised. Smaller
        t_start = larger deviation from the anchor.
        """
        device = device or next(self.parameters()).device
        temperature = float(temperature)
        if loss is not None:
            loss = loss.to(device).view(n, 1)
        if quantile_label is not None:
            quantile_label = quantile_label.to(device).long()
        if self.use_param_cond and params is not None:
            params = params.to(device).view(n, self.param_dim)

        if init_state is not None and t_start > 0.0:
            t0 = min(float(t_start), 0.95)
            anchor = init_state.to(device).view(n, 1, self.resolution).float()
            x = (1.0 - t0) * self._base_noise(n, device, temperature) + t0 * anchor
        else:
            t0 = 0.0
            x = self._base_noise(n, device, temperature)

        p = temperature * torch.randn(n, self.param_dim, device=device) if self.generate_params else None
        n_int = max(1, int(round((1.0 - t0) * self.steps)))
        dt = (1.0 - t0) / n_int
        for i in range(n_int):
            t = torch.full((n,), t0 + i * dt, device=device, dtype=torch.float32)
            state_in = p if self.generate_params else params
            v, pv = self._cfg_output(x, t, loss, state_in, quantile_label)
            x = x + dt * v
            if self.generate_params:
                p = p + dt * pv
        x = x.clamp(-6.0, 6.0)
        if self.generate_params:
            p = p.clamp(-1.5, 1.5)
        return x, p
