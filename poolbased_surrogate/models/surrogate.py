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
        padding_mode: str = "circular",
    ):
        super().__init__()
        ensure_al4pde_paths()
        from al4pde.modules.unet_cond_1d import Unet1D
        self.difference_weight = float(difference_weight)
        self.net = Unet1D(
            padding_mode=padding_mode,
            n_input_scalar_components=1,
            n_input_vector_components=0,
            n_output_scalar_components=1,
            n_output_vector_components=0,
            time_history=1,
            time_future=1,
            hidden_channels=int(hidden_channels),
            activation="gelu",
            norm=True,
            ch_mults=[1, 2, 2, 4],
            is_attn=[False, False, False, False],
            mid_attn=False,
            n_blocks=2,
            param_conditioning=f"scalar_{int(param_dim)}" if param_dim > 0 else None,
            use_scale_shift_norm=False,
            use1x1=False,
            n_dims=1,
            features="last_layer",
        )
        self.param_dim = param_dim

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            state = state[:, None, :] # [B, 1, X]
        if params.ndim == 1:
            params = params[:, None]
        out = self.net(state[:, None, :, :], z=params, time=None)
        delta = out[:, 0, :, :]
        return state + self.difference_weight * delta


class ExactAL4PDEUnet2D(nn.Module):
    """Native AL4PDE 2D Conditioned Unet wrapper."""
    def __init__(
        self,
        param_dim: int = 2,
        hidden_channels: int = 64,
        difference_weight: float = 0.3,
        padding_mode: str = "circular",
    ):
        super().__init__()
        ensure_al4pde_paths()
        from al4pde.modules.unet_cond_2d import Unet2D
        self.difference_weight = float(difference_weight)
        self.net = Unet2D(
            padding_mode=padding_mode,
            n_input_scalar_components=0,
            n_input_vector_components=1,
            n_output_scalar_components=0,
            n_output_vector_components=1,
            time_history=1,
            time_future=1,
            hidden_channels=int(hidden_channels),
            activation="gelu",
            norm=True,
            ch_mults=[1, 2, 2, 4],
            is_attn=[False, False, False, False],
            mid_attn=False,
            n_blocks=2,
            param_conditioning=f"scalar_{int(param_dim)}" if param_dim > 0 else None,
            use_scale_shift_norm=False,
            use1x1=False,
            features="last_layer",
        )
        self.param_dim = param_dim

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, None, :, :] # [B, C=2, H, W]
        if params.ndim == 1:
            params = params[:, None]
        out = self.net(state[:, None, :, :, :], z=params, time=None)
        delta = out[:, 0, :, :, :]
        return state + self.difference_weight * delta



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
