"""Config loading and small shared helpers for the loss-generator lab."""
from __future__ import annotations

import json
import numpy as np
import torch
from dataclasses import fields
from pathlib import Path
from typing import Any
from ..config import DDPMConfig, ExperimentConfig, PDEConfig, PoolConfig, SurrogateConfig, ValidationConfig, WandbConfig


def choose_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dataclass_from_dict(cls, data: dict[str, Any]):
    obj = cls()
    field_names = {f.name for f in fields(cls)}
    for key, value in (data or {}).items():
        if key in field_names:
            setattr(obj, key, value)
    return obj


def config_from_resolved_dict(data: dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    for key, value in (data or {}).items():
        if key == "pde":
            cfg.pde = _dataclass_from_dict(PDEConfig, value)
            if cfg.pde.viscosity_range is not None:
                cfg.pde.viscosity_range = tuple(cfg.pde.viscosity_range)
        elif key == "pool":
            cfg.pool = _dataclass_from_dict(PoolConfig, value)
        elif key == "surrogate":
            cfg.surrogate = _dataclass_from_dict(SurrogateConfig, value)
        elif key == "ddpm":
            cfg.ddpm = _dataclass_from_dict(DDPMConfig, value)
        elif key == "validation":
            cfg.validation = _dataclass_from_dict(ValidationConfig, value)
        elif key == "wandb":
            cfg.wandb = _dataclass_from_dict(WandbConfig, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def load_resolved_config(path: Path) -> ExperimentConfig:
    return config_from_resolved_dict(json.loads(path.read_text()))


def loss_norm_constants(losses: np.ndarray) -> tuple[float, float]:
    """The (q05, q95) constants used by train.normalize_losses.

    Persisted at dataset-build/training time so scoring can renormalize GENERATED
    losses onto the same scale that defined the quantile edges, instead of
    renormalizing by the generated batch's own quantiles (which breaks calibration
    whenever the generated loss distribution differs from the training one).
    """
    losses = np.asarray(losses, dtype=np.float32).reshape(-1)
    return float(np.quantile(losses, 0.05)), float(np.quantile(losses, 0.95))


def normalize_losses_with(losses: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """train.normalize_losses with externally supplied (training-time) constants."""
    losses = np.asarray(losses, dtype=np.float32)
    return ((losses - lo) / (hi - lo + 1e-8)).clip(0.0, 1.5).astype(np.float32)


def assign_loss_quantile_bins(losses: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_labels, bin_edges) where labels ∈ {0, …, n_bins-1}."""
    losses = np.asarray(losses, dtype=np.float32).reshape(-1)
    edges = np.quantile(losses, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = edges[-1] * (1 + 1e-6)  # ensure max sample falls in last bin
    labels = np.digitize(losses, edges[1:], right=False).clip(0, n_bins - 1).astype(np.int8)
    return labels, edges.astype(np.float32)


def stratified_split(losses: np.ndarray, seed: int, val_fraction: float, test_fraction: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    losses = np.asarray(losses).reshape(-1)
    split = np.zeros(len(losses), dtype=np.int8)
    if len(losses) == 0:
        return split
    edges = np.unique(np.quantile(losses, np.linspace(0.0, 1.0, 11)))
    if len(edges) <= 2:
        bins = np.zeros(len(losses), dtype=np.int64)
    else:
        bins = np.digitize(losses, edges[1:-1], right=True)
    for bin_id in np.unique(bins):
        idx = np.flatnonzero(bins == bin_id)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_fraction))
        n_val = int(round(len(idx) * val_fraction))
        split[idx[:n_test]] = 2
        split[idx[n_test : n_test + n_val]] = 1
    return split
