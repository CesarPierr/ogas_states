from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PDEConfig:
    name: str = "burgers"
    backend: str = "al4pde"
    resolution: int = 256
    dt: float = 2e-4
    domain_length: float = 1.0
    viscosity_range: tuple[float, float] = (0.003, 0.03)
    param_ranges: list[list[float]] | None = None
    param_log_scale: list[bool] | None = None
    ic: dict[str, Any] = field(default_factory=lambda: {"modes": 5, "amplitude": 1.0})


@dataclass
class PoolConfig:
    rounds: int = 8
    trajectory_steps: int = 75
    n_trajectories: int = 64
    uniform_fraction: float = 0.25


@dataclass
class SurrogateConfig:
    model: str = "al4pde_unet1d"
    hidden: int = 64
    depth: int = 3
    ensemble_size: int = 1
    epochs_per_round: int = 20
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-5
    difference_weight: float = 0.3


@dataclass
class DDPMConfig:
    enabled: bool = True
    mode: str = "conditional_loss"
    param_mode: str = "none"
    steps: int = 64
    train_epochs: int = 10
    batch_size: int = 128
    lr: float = 3e-4
    hidden: int = 64
    loss_condition_scale: float = 1.0
    param_loss_weight: float = 1.0


@dataclass
class ValidationConfig:
    n_trajectories: int = 128
    trajectory_steps: int = 99
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    rollout_steps: int = 50
    path: str | None = None
    epoch_n_trajectories: int = 128


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "poolbased-surrogate-1d"
    group: str = "default"


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs/default"
    pde: PDEConfig = field(default_factory=PDEConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _merge_dataclass(cls, data: dict[str, Any]):
    base = cls()
    for key, value in data.items():
        if not hasattr(base, key):
            raise ValueError(f"Unknown config key {cls.__name__}.{key}")
        setattr(base, key, value)
    return base


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = ExperimentConfig()
    for key, value in raw.items():
        if key == "pde":
            cfg.pde = _merge_dataclass(PDEConfig, value or {})
            cfg.pde.viscosity_range = tuple(cfg.pde.viscosity_range)
        elif key == "pool":
            cfg.pool = _merge_dataclass(PoolConfig, value or {})
        elif key == "surrogate":
            cfg.surrogate = _merge_dataclass(SurrogateConfig, value or {})
        elif key == "ddpm":
            cfg.ddpm = _merge_dataclass(DDPMConfig, value or {})
        elif key == "validation":
            cfg.validation = _merge_dataclass(ValidationConfig, value or {})
        elif key == "wandb":
            cfg.wandb = _merge_dataclass(WandbConfig, value or {})
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown config key {key}")
    return cfg
