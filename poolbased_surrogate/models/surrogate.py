from __future__ import annotations

import torch
from torch import nn

from ..al4pde_bridge import ExactAL4PDEUnet1D


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


def build_surrogate(
    resolution: int,
    hidden: int,
    depth: int,
    ensemble_size: int,
    param_dim: int = 1,
    model_name: str = "al4pde_unet1d",
    difference_weight: float = 0.3,
) -> EnsembleSurrogate:
    models = []
    for _ in range(max(1, int(ensemble_size))):
        if model_name == "conv_unet":
            models.append(
                ConvUNet1D(
                    resolution=resolution,
                    param_dim=param_dim,
                    hidden=hidden,
                    depth=depth,
                )
            )
        elif model_name == "al4pde_unet1d":
            models.append(
                ExactAL4PDEUnet1D(
                    param_dim=param_dim,
                    hidden_channels=hidden,
                    difference_weight=difference_weight,
                )
            )
        else:
            raise ValueError(f"Unknown surrogate model {model_name}")
    return EnsembleSurrogate(models)
