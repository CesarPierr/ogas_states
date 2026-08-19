"""
Surrogate Neural Operator Models: 1D & 2D Architecture
======================================================
Provides:
- ExactAL4PDEUnet1D: Standard 1D conditional Unet from AL4PDE benchmark.
- ExactAL4PDEUnet2D: Standard 2D conditional Unet from AL4PDE benchmark.
- EnsembleSurrogate: Ensemble wrapper with vectorized uncertainty estimation.
"""
from __future__ import annotations
import torch
from torch import nn
from typing import Sequence

from ..pde import ensure_al4pde_paths


class ExactAL4PDEUnet1D(nn.Module):
    """Native AL4PDE 1D Conditioned Unet wrapper."""
    def __init__(
        self,
        param_dim: int = 1,
        hidden_channels: int = 64,
        difference_weight: float = 0.3,
        num_layers: int = 5,
        n_channels: int = 1,
    ):
        super().__init__()
        ensure_al4pde_paths()
        from al4pde.modules.unet_cond_1d import Unet1D
        self.net = Unet1D(
            dim=1,
            num_channels=n_channels,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            use_conditioning=True,
            param_dim=param_dim,
            difference_weight=difference_weight,
        )
        self.param_dim = param_dim

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            state = state[:, None, :]
        if state.ndim == 3:
            state = state.unsqueeze(1) # [B, T=1, C, N]
        if params.ndim == 1:
            params = params[:, None]
        out = self.net(state, z=params)
        return out.squeeze(1)


class ExactAL4PDEUnet2D(nn.Module):
    """Native AL4PDE 2D Conditioned Unet wrapper."""
    def __init__(
        self,
        param_dim: int = 2,
        hidden_channels: int = 64,
        difference_weight: float = 0.3,
        num_layers: int = 5,
        n_channels: int = 2,
    ):
        super().__init__()
        ensure_al4pde_paths()
        from al4pde.modules.unet_cond_2d import Unet2D
        self.net = Unet2D(
            dim=2,
            num_channels=n_channels,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            use_conditioning=True,
            param_dim=param_dim,
            difference_weight=difference_weight,
        )
        self.param_dim = param_dim

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if state.ndim == 4:
            state = state.unsqueeze(1) # [B, T=1, C=2, H, W]
        if params.ndim == 1:
            params = params[:, None]
        out = self.net(state, z=params)
        return out.squeeze(1)


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
        if state.ndim == 2:
            state = state[:, None, :]
        if params.ndim == 1:
            params = params[:, None]
        cond = params[:, :, None].expand(-1, -1, state.shape[-1])
        x = torch.cat([state, cond], dim=1)
        delta = self.out(self.blocks(self.in_proj(x)))
        return state + delta


class EnsembleSurrogate(nn.Module):
    """Ensemble of surrogate models with vectorized uncertainty computation."""
    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(list(models))

    @property
    def n_models(self) -> int:
        return len(self.models)

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        preds = [m(state, params) for m in self.models]
        return torch.stack(preds, dim=0).mean(dim=0)

    @torch.no_grad()
    def uncertainty(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([m(state, params) for m in self.models], dim=0)
        return preds.var(dim=0).mean(dim=tuple(range(1, preds.ndim - 1)))


def build_surrogate(
    resolution: int,
    hidden: int,
    depth: int,
    ensemble_size: int,
    param_dim: int = 1,
    model_name: str = "al4pde_unet1d",
    difference_weight: float = 0.3,
    spatial_dim: int = 1,
) -> EnsembleSurrogate:
    models = []
    for _ in range(max(1, int(ensemble_size))):
        if spatial_dim == 2 or model_name == "al4pde_unet2d":
            models.append(
                ExactAL4PDEUnet2D(
                    param_dim=param_dim,
                    hidden_channels=hidden,
                    difference_weight=difference_weight,
                )
            )
        elif model_name == "conv_unet":
            models.append(
                ConvUNet1D(
                    resolution=resolution,
                    param_dim=param_dim,
                    hidden=hidden,
                    depth=depth,
                )
            )
        else: # al4pde_unet1d
            models.append(
                ExactAL4PDEUnet1D(
                    param_dim=param_dim,
                    hidden_channels=hidden,
                    difference_weight=difference_weight,
                )
            )
    return EnsembleSurrogate(models)
