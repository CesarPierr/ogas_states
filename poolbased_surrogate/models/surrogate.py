from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2, padding_mode="circular"),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2, padding_mode="circular"),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x + self.net(x))


class ConvUNet1D(nn.Module):
    def __init__(self, resolution: int, param_dim: int = 1, hidden: int = 64, depth: int = 3):
        super().__init__()
        self.resolution = resolution
        self.param_dim = param_dim
        self.in_proj = nn.Conv1d(1 + param_dim, hidden, 1)
        self.blocks = nn.Sequential(*[ConvBlock(hidden) for _ in range(depth)])
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        cond = params[:, :, None].expand(-1, -1, state.shape[-1])
        x = torch.cat([state, cond], dim=1)
        delta = self.out(self.blocks(self.in_proj(x)))
        return state + delta


class EnsembleSurrogate(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    @property
    def n_models(self) -> int:
        return len(self.models)

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        preds = [m(state, params) for m in self.models]
        return torch.stack(preds, dim=0).mean(dim=0)

    @torch.no_grad()
    def uncertainty(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([m(state, params) for m in self.models], dim=0)
        return preds.var(dim=0).mean(dim=(1, 2))


def build_surrogate(resolution: int, hidden: int, depth: int, ensemble_size: int) -> EnsembleSurrogate:
    models = [
        ConvUNet1D(resolution=resolution, hidden=hidden, depth=depth)
        for _ in range(max(1, int(ensemble_size)))
    ]
    return EnsembleSurrogate(models)
