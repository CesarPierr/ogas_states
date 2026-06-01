from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .config import DDPMConfig, ExperimentConfig, PDEConfig, PoolConfig, SurrogateConfig, ValidationConfig, WandbConfig
from .data import TransitionPool
from .models.ddpm import DDPM1D, FlowMatching1D
from .models.surrogate import build_surrogate
from .pde import PDE1D
from .train import compute_transition_losses, normalize_losses, transform_transition_losses

# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Dataset utilities
# ──────────────────────────────────────────────────────────────────────────────

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


def assign_loss_quantile_bins(losses: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_labels, bin_edges) where labels ∈ {0, …, n_bins-1}."""
    losses = np.asarray(losses, dtype=np.float32).reshape(-1)
    edges = np.quantile(losses, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = edges[-1] * (1 + 1e-6)  # ensure max sample falls in last bin
    labels = np.digitize(losses, edges[1:], right=False).clip(0, n_bins - 1).astype(np.int8)
    return labels, edges.astype(np.float32)


def _validation_state_stats(validation_path: Path | None, fallback_arrays: list[np.ndarray]) -> tuple[float, float]:
    if validation_path is not None and validation_path.exists():
        with np.load(validation_path) as payload:
            trajectories = payload["trajectories"]
            return float(trajectories.mean()), float(trajectories.std() + 1e-8)
    merged = np.concatenate([a.reshape(-1) for a in fallback_arrays], axis=0)
    return float(merged.mean()), float(merged.std() + 1e-8)


def _recompute_losses_with_surrogate(
    states: np.ndarray,
    params: np.ndarray,
    next_states: np.ndarray,
    surrogate_path: Path,
    cfg,
    batch_size: int = 256,
    device_name: str = "auto",
) -> np.ndarray:
    """Re-evaluate transition losses using the final trained surrogate."""
    from .models.surrogate import build_surrogate
    from .data import TransitionPool
    device = choose_device(device_name)
    surrogate = build_surrogate(
        resolution=cfg.pde.resolution,
        hidden=cfg.surrogate.hidden,
        depth=cfg.surrogate.depth,
        ensemble_size=cfg.surrogate.ensemble_size,
        param_dim=len(cfg.pde.param_ranges),
        model_name=cfg.surrogate.model,
        difference_weight=cfg.surrogate.difference_weight,
    ).to(device)
    surrogate.load_state_dict(torch.load(surrogate_path, map_location=device, weights_only=False))
    pool = TransitionPool(states=states, params=params, next_states=next_states)
    return compute_transition_losses(surrogate, pool, batch_size=batch_size, device=device)


def build_loss_dataset_from_run(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    round_id: int = 0,
    split_seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    n_quantile_bins: int = 10,
    recompute_with_final_surrogate: bool = False,
    device_name: str = "auto",
) -> Path:
    """Build a generator training dataset from one round of an AL run.

    Args:
        recompute_with_final_surrogate: If True, ignore the per-round losses stored in the
            pool file and recompute them using surrogate.pt (the final model, trained for all
            rounds).  This gives much cleaner quantile labels: at round 0 the surrogate has
            only seen 10 epochs and its losses have a p99/p50 ratio of ~114×; the final
            surrogate gives a ratio of ~11× (10× tighter), producing meaningful difficulty
            ordering for the quantile conditioning.
    """
    run_dir = Path(run_dir)
    pool_path = run_dir / f"pool_round_{round_id}.npz"
    config_path = run_dir / "config.resolved.json"
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing {pool_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    cfg = load_resolved_config(config_path)
    with np.load(pool_path) as pool:
        states = pool["states"].astype(np.float32)
        params = pool["params"].astype(np.float32)
        next_states = pool["next_states"].astype(np.float32)
        if "losses" not in pool:
            raise ValueError(f"{pool_path} does not contain post-training losses.")
        losses = pool["losses"].astype(np.float32).reshape(-1)
        pretrain_losses = pool["pretrain_losses"].astype(np.float32).reshape(-1) if "pretrain_losses" in pool else np.full_like(losses, np.nan)
        source = pool["source"].astype(np.int8).reshape(-1) if "source" in pool else np.zeros(len(losses), dtype=np.int8)

    surrogate_path = run_dir / "surrogate.pt"
    if recompute_with_final_surrogate:
        if not surrogate_path.exists():
            raise FileNotFoundError(f"Missing final surrogate at {surrogate_path}")
        print(f"Recomputing losses with final surrogate: {surrogate_path}", flush=True)
        original_losses = losses.copy()
        losses = _recompute_losses_with_surrogate(
            states, params, next_states, surrogate_path, cfg,
            batch_size=cfg.surrogate.batch_size, device_name=device_name,
        )
        print(f"  Round-{round_id} losses: p50={np.quantile(original_losses,0.5):.4f}  p99/p50={np.quantile(original_losses,0.99)/np.quantile(original_losses,0.5):.1f}x", flush=True)
        print(f"  Final-surrogate losses: p50={np.quantile(losses,0.5):.4f}  p99/p50={np.quantile(losses,0.99)/np.quantile(losses,0.5):.1f}x", flush=True)

    validation_path = Path(cfg.validation.path) if cfg.validation.path else None
    state_mean, state_std = _validation_state_stats(validation_path, [states, next_states])
    pde = PDE1D(cfg.pde)
    param_min = np.array([lo for lo, _ in pde.param_ranges], dtype=np.float32)
    param_max = np.array([hi for _, hi in pde.param_ranges], dtype=np.float32)
    split = stratified_split(losses, seed=split_seed, val_fraction=val_fraction, test_fraction=test_fraction)
    loss_norm = normalize_losses(losses)
    loss_norm_rmse = normalize_losses(transform_transition_losses(losses, "rmse"))
    loss_quantile_bins, quantile_edges = assign_loss_quantile_bins(losses, n_quantile_bins)

    if output_path is None:
        suffix = "_final_surrogate" if recompute_with_final_surrogate else ""
        output_path = run_dir / f"loss_generator_round{round_id}{suffix}.npz"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_dir": str(run_dir),
        "round_id": int(round_id),
        "recomputed_with_final_surrogate": recompute_with_final_surrogate,
        "config_path": str(config_path),
        "surrogate_path": str(surrogate_path),
        "validation_path": str(validation_path) if validation_path else None,
        "n_samples": int(len(losses)),
        "n_quantile_bins": n_quantile_bins,
        "state_mean": state_mean,
        "state_std": state_std,
        "loss_min": float(np.min(losses)),
        "loss_mean": float(np.mean(losses)),
        "loss_p50": float(np.quantile(losses, 0.5)),
        "loss_p90": float(np.quantile(losses, 0.9)),
        "loss_p95": float(np.quantile(losses, 0.95)),
        "loss_p99": float(np.quantile(losses, 0.99)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    np.savez_compressed(
        output_path,
        states=states,
        params=params,
        next_states=next_states,
        losses=losses,
        loss_norm=loss_norm,
        loss_norm_rmse=loss_norm_rmse,
        loss_quantile_bins=loss_quantile_bins,
        quantile_edges=quantile_edges,
        pretrain_losses=pretrain_losses,
        source=source,
        split=split,
        state_mean=np.array(state_mean, dtype=np.float32),
        state_std=np.array(state_std, dtype=np.float32),
        param_min=param_min,
        param_max=param_max,
        metadata=json.dumps(metadata, indent=2),
        resolved_config=json.dumps(json.loads(config_path.read_text())),
    )
    (output_path.with_suffix(".metadata.json")).write_text(json.dumps(metadata, indent=2))
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# State quality metrics
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_l2_stats(states: np.ndarray, max_points: int = 256, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    states = np.asarray(states, dtype=np.float32)
    if len(states) < 2:
        return {"pairwise_l2_mean": 0.0, "pairwise_l2_p10": 0.0, "pairwise_l2_p90": 0.0}
    idx = rng.choice(len(states), size=min(max_points, len(states)), replace=False)
    flat = states[idx].reshape(len(idx), -1)
    diff = flat[:, None, :] - flat[None, :, :]
    dist = np.sqrt(np.mean(diff * diff, axis=-1))
    tri = dist[np.triu_indices(len(idx), k=1)]
    return {
        "pairwise_l2_mean": float(np.mean(tri)),
        "pairwise_l2_p10": float(np.quantile(tri, 0.1)),
        "pairwise_l2_p90": float(np.quantile(tri, 0.9)),
    }


def state_quality_metrics(prefix: str, states: np.ndarray) -> dict[str, float]:
    states = np.asarray(states, dtype=np.float32)
    finite = np.isfinite(states)
    metrics = {
        f"{prefix}/finite_ratio": float(finite.mean()),
        f"{prefix}/value_mean": float(np.nanmean(states)),
        f"{prefix}/value_std": float(np.nanstd(states)),
        f"{prefix}/abs_p95": float(np.nanquantile(np.abs(states), 0.95)),
        f"{prefix}/abs_p99": float(np.nanquantile(np.abs(states), 0.99)),
        f"{prefix}/total_variation_mean": float(np.nanmean(np.sum(np.abs(np.diff(states, axis=-1)), axis=-1))),
        f"{prefix}/roughness_mean": float(np.nanmean(np.mean(np.diff(states, n=2, axis=-1) ** 2, axis=-1))),
    }
    fft = np.fft.rfft(np.nan_to_num(states.reshape(len(states), -1)), axis=-1)
    power = np.abs(fft) ** 2
    cutoff = max(1, int(power.shape[-1] * 0.25))
    metrics[f"{prefix}/highfreq_energy_ratio_mean"] = float(
        np.mean(power[:, cutoff:].sum(axis=1) / (power.sum(axis=1) + 1e-8))
    )
    metrics.update({f"{prefix}/{k}": v for k, v in pairwise_l2_stats(states).items()})
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Spectral + distribution realism metrics
# ──────────────────────────────────────────────────────────────────────────────

def psd_similarity_metrics(prefix: str, states: np.ndarray, ref_states: np.ndarray) -> dict[str, float]:
    """Compare mean log power-spectrum of generated states to a reference (e.g. validation set).

    A perfect generator reproduces the real KS spectral signature; psd_log_l2 → 0.
    High psd_high_freq_ratio means generated states have too much high-frequency energy.
    """
    s = np.nan_to_num(np.asarray(states, dtype=np.float32)).reshape(len(states), -1)
    r = np.nan_to_num(np.asarray(ref_states, dtype=np.float32)).reshape(len(ref_states), -1)
    fft_s = np.fft.rfft(s, axis=-1)
    fft_r = np.fft.rfft(r, axis=-1)
    psd_s = np.mean(np.abs(fft_s) ** 2, axis=0)  # (L//2+1,)
    psd_r = np.mean(np.abs(fft_r) ** 2, axis=0)
    log_diff = np.log(psd_s + 1e-8) - np.log(psd_r + 1e-8)
    hf = max(1, len(psd_s) // 4)  # upper 75% of spectrum = high-freq
    return {
        f"{prefix}/psd_log_l2": float(np.sqrt(np.mean(log_diff ** 2))),
        f"{prefix}/psd_log_max_err": float(np.max(np.abs(log_diff))),
        f"{prefix}/psd_hf_ratio": float(
            (psd_s[hf:].sum() + 1e-8) / (psd_r[hf:].sum() + 1e-8)
        ),
    }


def amplitude_distribution_metrics(prefix: str, states: np.ndarray, ref_states: np.ndarray) -> dict[str, float]:
    """Compare per-state amplitude (std) distribution: generated vs reference.

    amplitude_ratio close to 1 = right energy; amplitude_wd near 0 = distribution matches.
    """
    from scipy.stats import wasserstein_distance
    s = np.nan_to_num(np.asarray(states, dtype=np.float32)).reshape(len(states), -1)
    r = np.nan_to_num(np.asarray(ref_states, dtype=np.float32)).reshape(len(ref_states), -1)
    s_std = np.std(s, axis=-1)
    r_std = np.std(r, axis=-1)
    return {
        f"{prefix}/amplitude_std_mean": float(np.mean(s_std)),
        f"{prefix}/amplitude_ratio": float(np.mean(s_std) / (np.mean(r_std) + 1e-8)),
        f"{prefix}/amplitude_wd": float(wasserstein_distance(s_std, r_std)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Conditioning fidelity metrics
# ──────────────────────────────────────────────────────────────────────────────

def conditioning_fidelity_metrics(
    prefix: str,
    target_bins: np.ndarray,
    realized_bins: np.ndarray,
    n_quantiles: int,
) -> dict[str, float]:
    """Measure how well the quantile conditioning controls realized difficulty.

    Metrics:
      calibration_mae:       normalised mean absolute error between target and realised bin
                             (0 = perfect, 1 = always off by the full range)
      calibration_monotone:  fraction of adjacent bin pairs where E[realized|k+1] > E[realized|k]
                             (1 = strictly monotone conditioning, 0.5 = random)
      conditioning_mi_nats:  mutual information I(target; realised) in nats
                             (log(n_quantiles) ≈ 2.3 for n=10 = perfect; 0 = no signal)
      per_bin/k/mean_realized: for each target bin k, mean realised bin (calibration curve)
    """
    target_bins = np.asarray(target_bins, dtype=np.int64).clip(0, n_quantiles - 1)
    realized_bins = np.asarray(realized_bins, dtype=np.int64).clip(0, n_quantiles - 1)

    per_bin_mean: list[float] = []
    for k in range(n_quantiles):
        mask = target_bins == k
        per_bin_mean.append(float(np.mean(realized_bins[mask])) if mask.any() else float("nan"))

    valid = [(k, v) for k, v in enumerate(per_bin_mean) if math.isfinite(v)]
    cal_mae = float(np.mean([abs(v - k) / max(n_quantiles - 1, 1) for k, v in valid])) if valid else float("nan")

    valid_means = [v for v in per_bin_mean if math.isfinite(v)]
    if len(valid_means) > 1:
        n_pairs = len(valid_means) - 1
        monotone_frac = float(sum(valid_means[i] < valid_means[i + 1] for i in range(n_pairs)) / n_pairs)
    else:
        monotone_frac = float("nan")

    # Mutual information via empirical joint distribution
    joint = np.zeros((n_quantiles, n_quantiles), dtype=np.float64)
    for t, r in zip(target_bins, realized_bins):
        joint[t, r] += 1.0
    joint /= joint.sum() + 1e-12
    p_t = joint.sum(axis=1, keepdims=True) + 1e-12
    p_r = joint.sum(axis=0, keepdims=True) + 1e-12
    mi = float(np.sum(joint * np.log((joint + 1e-12) / (p_t * p_r))))

    out: dict[str, float] = {
        f"{prefix}/calibration_mae": cal_mae,
        f"{prefix}/calibration_monotone_frac": monotone_frac,
        f"{prefix}/conditioning_mi_nats": mi,
    }
    for k, v in enumerate(per_bin_mean):
        out[f"{prefix}/per_bin/{k}/mean_realized"] = v
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Intra-bin diversity
# ──────────────────────────────────────────────────────────────────────────────

def intra_bin_diversity_metrics(
    prefix: str,
    states: np.ndarray,
    bin_labels: np.ndarray,
    n_quantiles: int,
    max_per_bin: int = 64,
    seed: int = 0,
) -> dict[str, float]:
    """Mean pairwise L2 distance within each quantile bin.

    Low intra-bin diversity → the generator collapses to one output per quantile bin.
    We want intra-bin diversity to be high (states are unique within each hardness class).
    """
    rng = np.random.default_rng(seed)
    flat = np.nan_to_num(np.asarray(states, dtype=np.float32)).reshape(len(states), -1)
    bin_labels = np.asarray(bin_labels, dtype=np.int64).clip(0, n_quantiles - 1)
    per_bin_div: list[float] = []
    for k in range(n_quantiles):
        idx = np.flatnonzero(bin_labels == k)
        if len(idx) < 2:
            continue
        if len(idx) > max_per_bin:
            idx = rng.choice(idx, max_per_bin, replace=False)
        sub = flat[idx]
        diff = sub[:, None, :] - sub[None, :, :]
        dist = np.sqrt(np.mean(diff ** 2, axis=-1))
        tri = dist[np.triu_indices(len(idx), k=1)]
        per_bin_div.append(float(np.mean(tri)))
    return {
        f"{prefix}/intra_bin_diversity_mean": float(np.mean(per_bin_div)) if per_bin_div else float("nan"),
        f"{prefix}/intra_bin_diversity_min": float(np.min(per_bin_div)) if per_bin_div else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Param scaling helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scale_params(params: np.ndarray, param_min: np.ndarray, param_max: np.ndarray) -> np.ndarray:
    return (2.0 * (params - param_min) / (param_max - param_min + 1e-8) - 1.0).astype(np.float32)


def _unscale_params(params: np.ndarray, param_min: np.ndarray, param_max: np.ndarray) -> np.ndarray:
    return (0.5 * (params + 1.0) * (param_max - param_min) + param_min).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Generator factory
# ──────────────────────────────────────────────────────────────────────────────

def make_generator(
    generator: str,
    resolution: int,
    hidden: int,
    steps: int,
    loss_conditional: bool,
    param_dim: int,
    param_mode: str,
    residual_blocks: int,
    kernel_size: int,
    n_quantiles: int = 0,
    quant_embed_dim: int = 0,
):
    cls = FlowMatching1D if generator.lower() in ("flow", "flow_matching", "fm") else DDPM1D
    return cls(
        resolution=resolution,
        hidden=hidden,
        steps=steps,
        loss_conditional=loss_conditional,
        param_dim=param_dim,
        param_mode=param_mode,
        residual_blocks=residual_blocks,
        kernel_size=kernel_size,
        n_quantiles=n_quantiles,
        quant_embed_dim=quant_embed_dim,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _balanced_sampler_from_quantiles(quantile_labels: np.ndarray, n_quantiles: int) -> WeightedRandomSampler:
    """Sample each quantile bin with equal expected frequency per epoch."""
    bin_counts = np.bincount(quantile_labels.astype(np.int64), minlength=n_quantiles).astype(np.float64)
    bin_counts = np.maximum(bin_counts, 1)
    weights = (1.0 / bin_counts[quantile_labels.astype(np.int64)]).astype(np.float32)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def train_loss_generator(
    dataset_path: str | Path,
    output_dir: str | Path,
    generator: str = "flow_matching",
    param_mode: str = "condition",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 2e-4,
    hidden: int = 128,
    steps: int = 16,
    residual_blocks: int = 2,
    kernel_size: int = 7,
    loss_metric: str = "rmse",
    n_quantiles: int = 10,
    quant_embed_dim: int = 0,
    seed: int = 0,
    device_name: str = "auto",
    eval_every: int = 0,
    eval_n_samples: int = 512,
    eval_solver_batch_size: int = 512,
    eval_jax_platform: str = "auto",
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    wandb_extra_config: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, float]]:
    """Train a state generator conditioned on quantile labels.

    States are z-normalised before training (mean/std from validation or pool)
    and denormalised back to physical units at sampling time.

    The conditioning is a discrete quantile label in {0, …, n_quantiles-1}
    routed through a learned embedding.  Training uses a balanced per-bin
    sampler so the model sees equal signal from every difficulty level.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device(device_name)

    with np.load(dataset_path, allow_pickle=False) as data:
        states = data["states"].astype(np.float32)
        params = data["params"].astype(np.float32)
        raw_losses = data["losses"].astype(np.float32).reshape(-1)
        split = data["split"].astype(np.int8)
        state_mean = float(data["state_mean"])
        state_std = float(data["state_std"])
        param_min = data["param_min"].astype(np.float32)
        param_max = data["param_max"].astype(np.float32)
        resolved_config = str(data["resolved_config"])
        # Re-assign quantile bins from the full dataset so edges are consistent,
        # unless n_quantiles is overridden/different.
        if "quantile_edges" in data and (len(data["quantile_edges"]) - 1) == n_quantiles:
            quantile_edges = data["quantile_edges"].astype(np.float32)
            quantile_bins_all = data["loss_quantile_bins"].astype(np.int8)
        else:
            quantile_bins_all, quantile_edges = assign_loss_quantile_bins(
                normalize_losses(transform_transition_losses(raw_losses, loss_metric)), n_quantiles
            )

    train_mask = split == 0
    # States are z-normalised: the generator trains in a unit-Gaussian-like space,
    # making it much easier to learn the diffusion/flow objective.
    state_norm = ((states[train_mask] - state_mean) / state_std).astype(np.float32)
    params_scaled = _scale_params(params[train_mask], param_min, param_max)
    quantile_train = quantile_bins_all[train_mask].astype(np.int64)

    dataset = TensorDataset(
        torch.from_numpy(state_norm),
        torch.from_numpy(quantile_train),
        torch.from_numpy(params_scaled),
    )
    # Balanced sampler: equal bin frequency so conditioning signal has uniform coverage.
    sampler = _balanced_sampler_from_quantiles(quantile_train, n_quantiles)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)

    model = make_generator(
        generator=generator,
        resolution=states.shape[-1],
        hidden=hidden,
        steps=steps,
        loss_conditional=True,
        param_dim=params.shape[1],
        param_mode=param_mode,
        residual_blocks=residual_blocks,
        kernel_size=kernel_size,
        n_quantiles=n_quantiles,
        quant_embed_dim=quant_embed_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []
    n_params = sum(p.numel() for p in model.parameters())
    train_start = time.perf_counter()

    # Optional wandb run for this single config
    wandb_run = None
    if wandb_project:
        try:
            import wandb  # noqa: PLC0415
            wandb_cfg = {
                "generator": generator,
                "param_mode": param_mode,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "hidden": hidden,
                "steps": steps,
                "residual_blocks": residual_blocks,
                "kernel_size": kernel_size,
                "loss_metric": loss_metric,
                "n_quantiles": n_quantiles,
                "quant_embed_dim": quant_embed_dim,
                "seed": seed,
                "n_params": n_params,
                "dataset_path": str(dataset_path),
                "eval_every": eval_every,
                "eval_n_samples": eval_n_samples,
            }
            if wandb_extra_config:
                wandb_cfg.update(wandb_extra_config)
            init_kwargs = {
                "project": wandb_project,
                "name": wandb_run_name,
                "group": wandb_group,
                "config": wandb_cfg,
                "dir": str(output_dir),
                "reinit": True,
            }
            if not os.environ.get("WANDB_API_KEY"):
                init_kwargs["anonymous"] = "allow"
            wandb_run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
            wandb_run.define_metric("epoch")
            wandb_run.define_metric("train/*", step_metric="epoch")
            wandb_run.define_metric("eval/*", step_metric="epoch")
        except Exception as e:  # noqa: BLE001
            print(f"  [wandb] init failed ({e}) — continuing without wandb", flush=True)
            wandb_run = None

    def _build_ckpt() -> dict:
        return {
            "model": model.state_dict(),
            "dataset_path": str(dataset_path),
            "resolved_config": resolved_config,
            "generator": generator,
            "param_mode": param_mode,
            "hidden": hidden,
            "steps": steps,
            "residual_blocks": residual_blocks,
            "kernel_size": kernel_size,
            "loss_metric": loss_metric,
            "n_quantiles": n_quantiles,
            "quant_embed_dim": quant_embed_dim,
            "quantile_edges": quantile_edges,
            "state_mean": state_mean,
            "state_std": state_std,
            "param_min": param_min,
            "param_max": param_max,
            "n_params": n_params,
        }

    ckpt_path = output_dir / "loss_generator.pt"

    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        model.train()
        losses = []
        for state_b, qbin_b, param_b in loader:
            opt.zero_grad(set_to_none=True)
            loss = model.training_loss(
                state_b.to(device),
                params=param_b.to(device) if model.use_param_cond else None,
                quantile_label=qbin_b.to(device),
                param_loss_weight=1.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        record = {
            "epoch": float(epoch),
            "train/loss": float(np.mean(losses)),
            "epoch_time_s": float(time.perf_counter() - epoch_t0),
            "wall_time_s": float(time.perf_counter() - train_start),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": record["train/loss"],
                "train/epoch_time_s": record["epoch_time_s"],
                "train/wall_time_s": record["wall_time_s"],
            })

        # Periodic full 5-axis validation eval
        do_eval = (
            eval_every > 0
            and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
        )
        if do_eval:
            eval_t0 = time.perf_counter()
            # Persist current model so score_generated_samples can load it.
            torch.save(_build_ckpt(), ckpt_path)
            eval_dir = output_dir / f"eval_ep{epoch + 1}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [eval@epoch={epoch + 1}] scoring (n_samples={eval_n_samples}) ...", flush=True)
            try:
                score = score_generated_samples(
                    checkpoint_path=ckpt_path,
                    dataset_path=dataset_path,
                    output_dir=eval_dir,
                    n_samples=eval_n_samples,
                    seed=seed,
                    device_name=device_name,
                    solver_batch_size=eval_solver_batch_size,
                    jax_platform=eval_jax_platform,
                )
                model.train()  # score_generated_samples puts model into eval mode through sampling
            except Exception as e:  # noqa: BLE001
                print(f"  [eval@epoch={epoch + 1}] FAILED: {e}", flush=True)
                score = {}
            eval_time = time.perf_counter() - eval_t0
            eval_entry = _summarize_eval_for_history(score, epoch=epoch + 1, eval_time_s=eval_time)
            eval_history.append(eval_entry)
            # Surface eval entry in history.json so wall-clock plots include it
            history[-1] = {**record, **{f"eval/{k}": v for k, v in eval_entry.items() if k != "epoch"}}
            print(json.dumps({"eval_summary": eval_entry}), flush=True)
            (output_dir / "eval_history.json").write_text(json.dumps(eval_history, indent=2))
            if wandb_run is not None:
                # Log the compact high-level summary metrics under eval/
                wandb_payload = {"epoch": epoch}
                for k, v in eval_entry.items():
                    if k == "epoch":
                        continue
                    try:
                        wandb_payload[f"eval/{k}"] = float(v)
                    except (TypeError, ValueError):
                        pass
                
                # Log selected key full metrics for top1 and uniform strategies to avoid metric clutter
                for k, v in score.items():
                    # Only keep top1 and uniform strategy metrics, and essential reference stats
                    if not (k.startswith("top1/") or k.startswith("uniform/")):
                        if k.startswith("reference_states/") or k.startswith("validation_states/") or k.startswith("reference/"):
                            if not any(p in k for p in ["total_variation_mean", "loss_mean"]):
                                continue
                        else:
                            continue
                    
                    # Filter out highly detailed, noisy, or redundant statistics
                    exclude_patterns = [
                        "per_bin", "states/mean", "states/std", "states/min", "states/max",
                        "states/abs_p", "states/pairwise_l2_", "states/highfreq_energy", "states/roughness",
                        "value_mean", "value_std", "target_quantile_mean", "realized_quantile_mean",
                        "exact_quantile_accuracy", "top_quantile_hit_rate"
                    ]
                    if any(p in k for p in exclude_patterns):
                        continue
                        
                    try:
                        # Log under eval/ to keep namespace shallow and clean
                        wandb_payload[f"eval/{k}"] = float(v)
                    except (TypeError, ValueError):
                        pass
                wandb_run.log(wandb_payload)

    total_train_time = time.perf_counter() - train_start

    # Convergence epoch (train loss): first epoch where loss ≤ 1.05× min(loss)
    all_losses = [h["train/loss"] for h in history]
    min_loss = min(all_losses)
    convergence_epoch = next((i for i, l in enumerate(all_losses) if l <= 1.05 * min_loss), epochs - 1)

    # Convergence epoch (validation): first eval-checkpoint whose combined_score
    # is within 5% of the best across all eval points.
    best_eval_epoch = -1
    convergence_epoch_eval = -1
    if eval_history:
        scored = [(int(e["epoch"]), float(e.get("combined_score", float("nan")))) for e in eval_history]
        scored = [(ep, s) for ep, s in scored if math.isfinite(s)]
        if scored:
            best_eval_epoch, best_score = max(scored, key=lambda x: x[1])
            threshold = 0.95 * best_score if best_score > 0 else best_score
            convergence_epoch_eval = next((ep for ep, s in scored if s >= threshold), best_eval_epoch)

    ckpt = {
        **_build_ckpt(),
        "history": history,
        "eval_history": eval_history,
        "total_train_time_s": total_train_time,
        "convergence_epoch": convergence_epoch,
        "convergence_epoch_eval": convergence_epoch_eval,
        "best_eval_epoch": best_eval_epoch,
    }
    torch.save(ckpt, ckpt_path)
    last = {
        **history[-1],
        "total_train_time_s": total_train_time,
        "convergence_epoch": float(convergence_epoch),
        "convergence_epoch_eval": float(convergence_epoch_eval),
        "best_eval_epoch": float(best_eval_epoch),
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    if eval_history:
        (output_dir / "eval_history.json").write_text(json.dumps(eval_history, indent=2))
    if wandb_run is not None:
        try:
            for k, v in last.items():
                try:
                    wandb_run.summary[f"final/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
            wandb_run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"  [wandb] finish failed: {e}", flush=True)
    return ckpt_path, last


def _summarize_eval_for_history(score: dict[str, float], epoch: int, eval_time_s: float) -> dict[str, float]:
    """Reduce a full score_metrics dict to a compact per-eval-epoch summary.

    Captures one row per axis so we can plot how realism / hardness / conditioning /
    diversity / speed evolve over training, and derive a single combined_score that
    drives convergence-epoch detection.
    """
    def _f(key: str, default: float = float("nan")) -> float:
        v = score.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    ref_tv = _f("validation_states/total_variation_mean", _f("reference_states/total_variation_mean", float("nan")))
    tv_top1 = _f("top1/states/total_variation_mean")
    tv_ratio = tv_top1 / ref_tv if math.isfinite(tv_top1) and math.isfinite(ref_tv) and ref_tv > 0 else float("nan")

    # Combined score: realism × hardness × conditioning. See analyze_sweep.recommend().
    if math.isfinite(tv_ratio) and tv_ratio <= 3.0:
        tv_score = math.exp(-0.5 * (tv_ratio - 1.0) ** 2)
    else:
        tv_score = 0.0
    loss_ratio_top1 = _f("top1/loss_vs_reference_ratio")
    ratio_norm = min(loss_ratio_top1 / 5.0, 1.0) if math.isfinite(loss_ratio_top1) else 0.0
    calib_mae = _f("top1/calibration_mae")
    calib_score = max(0.0, 1.0 - calib_mae) if math.isfinite(calib_mae) else 0.0
    combined = 0.4 * tv_score + 0.4 * ratio_norm + 0.2 * calib_score

    return {
        "epoch": float(epoch),
        "eval_time_s": float(eval_time_s),
        # realism
        "tv_top1": tv_top1,
        "tv_ratio": tv_ratio,
        "amplitude_ratio_top1": _f("top1/states/amplitude_ratio"),
        "psd_log_l2_top1": _f("top1/states/psd_log_l2"),
        # hardness
        "loss_ratio_top1": loss_ratio_top1,
        "loss_ratio_uniform": _f("uniform/loss_vs_reference_ratio"),
        # conditioning
        "calibration_mae_top1": calib_mae,
        "calibration_monotone_frac_top1": _f("top1/calibration_monotone_frac"),
        "conditioning_mi_uniform": _f("uniform/conditioning_mi_nats"),
        "rank_corr_uniform": _f("uniform/target_vs_realized_rank_corr"),
        # diversity
        "intra_bin_diversity_top1": _f("top1/states/intra_bin_diversity_mean"),
        # speed
        "gen_time_ms_top1": _f("top1/gen_time_per_sample_ms"),
        # combined
        "tv_score": tv_score,
        "ratio_norm": ratio_norm,
        "calib_score": calib_score,
        "combined_score": combined,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_loss_generator(
    checkpoint_path: str | Path,
    n_samples: int,
    quantile_labels: np.ndarray,
    params: np.ndarray,
    device_name: str = "auto",
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample states from the generator conditioned on quantile labels.

    Returns (states_physical, params_physical).
    states are returned in physical units (denormalised from z-norm training space).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = choose_device(device_name)
    param_min = np.asarray(checkpoint["param_min"], dtype=np.float32)
    param_max = np.asarray(checkpoint["param_max"], dtype=np.float32)
    state_mean = float(checkpoint["state_mean"])
    state_std = float(checkpoint["state_std"])
    n_quantiles = int(checkpoint.get("n_quantiles", 0))
    quant_embed_dim = int(checkpoint.get("quant_embed_dim", 0))
    resolution = int(json.loads(checkpoint["resolved_config"])["pde"]["resolution"])

    model = make_generator(
        generator=str(checkpoint["generator"]),
        resolution=resolution,
        hidden=int(checkpoint["hidden"]),
        steps=int(checkpoint["steps"]),
        loss_conditional=True,
        param_dim=len(param_min),
        param_mode=str(checkpoint["param_mode"]),
        residual_blocks=int(checkpoint["residual_blocks"]),
        kernel_size=int(checkpoint["kernel_size"]),
        n_quantiles=n_quantiles,
        quant_embed_dim=quant_embed_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    quantile_labels = np.asarray(quantile_labels, dtype=np.int64).reshape(-1)
    if len(quantile_labels) != n_samples:
        quantile_labels = np.resize(quantile_labels, n_samples)
    params = np.asarray(params, dtype=np.float32)
    if len(params) != n_samples:
        params = np.resize(params, (n_samples, params.shape[1]))
    params_scaled = _scale_params(params, param_min, param_max)

    ql_t = torch.from_numpy(quantile_labels).to(device)
    pt_t = torch.from_numpy(params_scaled).to(device) if model.use_param_cond else None
    gen_norm, gen_params_scaled = model.sample(
        n_samples, params=pt_t, device=device, temperature=temperature, quantile_label=ql_t
    )
    # Denormalise: states were z-normed during training, reverse that here.
    states = (gen_norm.cpu().numpy() * state_std + state_mean).astype(np.float32)
    if gen_params_scaled is not None:
        params = _unscale_params(gen_params_scaled.cpu().numpy(), param_min, param_max)
    return states, params.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Quantile sampling strategies
# ──────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _sampling_strategies(n_quantiles: int) -> dict[str, Any]:
    """Return a dict of strategy_name → callable(n, rng) → int array of quantile labels."""
    q = n_quantiles
    return {
        "top1":      lambda n, rng: np.full(n, q - 1, dtype=np.int64),
        "top2":      lambda n, rng: rng.choice([q - 2, q - 1], size=n),
        "top3":      lambda n, rng: rng.choice(np.arange(max(0, q - 3), q), size=n),
        "top5":      lambda n, rng: rng.choice(np.arange(max(0, q - 5), q), size=n),
        "top_half":  lambda n, rng: rng.integers(q // 2, q, size=n),
        "uniform":   lambda n, rng: rng.integers(0, q, size=n),
        "exp_bias":  lambda n, rng: rng.choice(q, size=n, p=_softmax(np.arange(q, dtype=float) * 0.5)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def _load_validation_ref_states(cfg, n_max: int = 1024, rng: np.random.Generator | None = None) -> np.ndarray | None:
    """Load a sample of validation states for spectral/amplitude reference comparisons."""
    val_path = getattr(cfg.validation, "path", None)
    if not val_path:
        return None
    val_path = Path(val_path)
    if not val_path.exists():
        return None
    try:
        with np.load(val_path, allow_pickle=False) as vd:
            trajs = vd["trajectories"].astype(np.float32)  # (N, T+1, 1, L)
        states_flat = trajs[:, 1:].reshape(-1, 1, trajs.shape[-1])  # skip t=0
        if rng is not None and len(states_flat) > n_max:
            idx = rng.choice(len(states_flat), n_max, replace=False)
            states_flat = states_flat[idx]
        return states_flat[:n_max]
    except Exception as e:
        print(f"  Warning: could not load validation states for realism metrics: {e}", flush=True)
        return None


def _pde_step_batch(pde, states: np.ndarray, params: np.ndarray, solver_batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(states), solver_batch_size):
        end = min(start + solver_batch_size, len(states))
        try:
            ns = pde.step(states[start:end, 0], params[start:end])[:, None, :]
        except Exception:
            ns = np.full((end - start, 1, states.shape[-1]), np.nan, dtype=np.float32)
        chunks.append(ns)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def score_generated_samples(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    n_samples: int = 1024,
    seed: int = 0,
    device_name: str = "auto",
    solver_batch_size: int = 512,
    temperature: float = 1.0,
    jax_platform: str = "auto",
) -> dict[str, float]:
    """Score a trained generator on five axes:

    1. Realism     — PSD similarity + amplitude distribution vs real KS validation states
    2. Hardness    — loss_vs_reference_ratio (> 1 = harder than uniform pool)
    3. Conditioning fidelity — calibration_mae, calibration_monotone_frac, conditioning_mi_nats
    4. Diversity   — pairwise L2 (global) + intra-bin diversity per quantile bin
    5. Speed       — gen_time_per_sample_ms (recorded at sampling time)
    """
    rng = np.random.default_rng(seed)
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(dataset_path, allow_pickle=False) as data:
        params_all = data["params"].astype(np.float32)
        states_all = data["states"].astype(np.float32)
        raw_losses = data["losses"].astype(np.float32)
        split = data["split"].astype(np.int8)
        resolved_config = json.loads(str(data["resolved_config"]))
        metadata = json.loads(str(data["metadata"]))
        quantile_edges = data["quantile_edges"].astype(np.float32) if "quantile_edges" in data else None

    device = choose_device(device_name)
    if jax_platform != "auto":
        os.environ["JAX_PLATFORMS"] = jax_platform
    elif device.type == "cpu" and not torch.cuda.is_available():
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    cfg = config_from_resolved_dict(resolved_config)
    pde = PDE1D(cfg.pde)
    holdout_idx = np.flatnonzero(split > 0)
    if len(holdout_idx) == 0:
        holdout_idx = np.arange(len(params_all))

    ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    n_quantiles = int(ckpt_meta.get("n_quantiles", 10))
    loss_metric = str(ckpt_meta.get("loss_metric", "rmse"))
    n_params_model = int(ckpt_meta.get("n_params", 0))

    if "quantile_edges" in ckpt_meta:
        quantile_edges = np.asarray(ckpt_meta["quantile_edges"], dtype=np.float32)
    if quantile_edges is None:
        _, quantile_edges = assign_loss_quantile_bins(
            normalize_losses(transform_transition_losses(raw_losses, loss_metric)), n_quantiles
        )

    surrogate_path = Path(metadata["surrogate_path"])
    surrogate = build_surrogate(
        resolution=cfg.pde.resolution,
        hidden=cfg.surrogate.hidden,
        depth=cfg.surrogate.depth,
        ensemble_size=cfg.surrogate.ensemble_size,
        param_dim=len(pde.param_ranges),
        model_name=cfg.surrogate.model,
        difference_weight=cfg.surrogate.difference_weight,
    ).to(device)
    surrogate.load_state_dict(torch.load(surrogate_path, map_location=device, weights_only=False))

    ref_sample_idx = rng.choice(len(holdout_idx), size=min(n_samples, len(holdout_idx)), replace=False)
    ref_states = states_all[holdout_idx[ref_sample_idx]]
    ref_losses = raw_losses[holdout_idx[rng.choice(len(holdout_idx), size=n_samples, replace=True)]]
    params_for_cond = params_all[holdout_idx[rng.choice(len(holdout_idx), size=n_samples, replace=True)]]

    # Real KS validation states for realism comparison (PSD, amplitude)
    val_ref_states = _load_validation_ref_states(cfg, n_max=n_samples, rng=rng)

    strategies = _sampling_strategies(n_quantiles)
    all_metrics: dict[str, float] = {
        "model/n_params": float(n_params_model),
        "model/n_quantiles": float(n_quantiles),
        "model/steps": float(ckpt_meta.get("steps", 0)),
        "model/hidden": float(ckpt_meta.get("hidden", 0)),
        "model/residual_blocks": float(ckpt_meta.get("residual_blocks", 0)),
        "model/quant_embed_dim": float(ckpt_meta.get("quant_embed_dim", 0)),
        "model/convergence_epoch": float(ckpt_meta.get("convergence_epoch", -1)),
        "model/total_train_time_s": float(ckpt_meta.get("total_train_time_s", float("nan"))),
        "reference/loss_mean": float(np.mean(ref_losses)),
        "reference/loss_p50": float(np.quantile(ref_losses, 0.5)),
        "reference/loss_p90": float(np.quantile(ref_losses, 0.9)),
        "reference/loss_p99": float(np.quantile(ref_losses, 0.99)),
    }
    all_metrics.update(state_quality_metrics("reference_states", ref_states))
    if val_ref_states is not None:
        all_metrics.update(state_quality_metrics("validation_states", val_ref_states))

    for strategy_name, sample_fn in strategies.items():
        target_quantiles = sample_fn(n_samples, rng).astype(np.int64)

        t0 = time.perf_counter()
        states, params = sample_loss_generator(
            checkpoint_path, n_samples=n_samples, quantile_labels=target_quantiles,
            params=params_for_cond, device_name=device_name, temperature=temperature,
        )
        gen_time_ms = (time.perf_counter() - t0) / max(n_samples, 1) * 1000.0

        next_states = _pde_step_batch(pde, states, params, solver_batch_size)
        pool = TransitionPool(states=states, params=params, next_states=next_states)
        gen_losses = compute_transition_losses(surrogate, pool, batch_size=cfg.surrogate.batch_size, device=device)

        finite = (
            np.isfinite(states).all(axis=(1, 2))
            & np.isfinite(next_states).all(axis=(1, 2))
            & np.isfinite(gen_losses)
        )

        m: dict[str, float] = {
            f"{strategy_name}/finite_ratio": float(finite.mean()),
            f"{strategy_name}/gen_time_per_sample_ms": float(gen_time_ms),
        }
        m.update(state_quality_metrics(f"{strategy_name}/states", states))

        # 1. Realism vs real KS validation states
        if val_ref_states is not None:
            m.update(psd_similarity_metrics(f"{strategy_name}/states", states, val_ref_states))
            m.update(amplitude_distribution_metrics(f"{strategy_name}/states", states, val_ref_states))

        if not finite.any():
            print(f"  WARNING: {strategy_name} — all {n_samples} samples invalid; skipping loss/conditioning metrics.", flush=True)
            for key in ("loss_mean", "loss_vs_reference_ratio", "target_vs_realized_rank_corr",
                        "calibration_mae", "calibration_monotone_frac", "conditioning_mi_nats",
                        "intra_bin_diversity_mean"):
                m[f"{strategy_name}/{key}"] = math.nan
            all_metrics.update(m)
            continue

        # 2. Hardness vs uniform reference
        gen_loss_mean = float(np.mean(gen_losses[finite]))
        ref_loss_mean = float(np.mean(ref_losses))
        m.update({
            f"{strategy_name}/loss_mean": gen_loss_mean,
            f"{strategy_name}/loss_p50": float(np.quantile(gen_losses[finite], 0.5)),
            f"{strategy_name}/loss_p90": float(np.quantile(gen_losses[finite], 0.9)),
            f"{strategy_name}/loss_p99": float(np.quantile(gen_losses[finite], 0.99)),
            f"{strategy_name}/loss_vs_reference_ratio": gen_loss_mean / (ref_loss_mean + 1e-12),
        })

        # 3. Conditioning fidelity
        realized_norm = normalize_losses(transform_transition_losses(gen_losses[finite], loss_metric))
        realized_quantiles = np.digitize(realized_norm, quantile_edges[1:], right=False).clip(0, n_quantiles - 1)
        target_q_finite = target_quantiles[finite]

        m[f"{strategy_name}/target_quantile_mean"] = float(np.mean(target_quantiles))
        m[f"{strategy_name}/realized_quantile_mean"] = float(np.mean(realized_quantiles))

        if len(realized_quantiles) > 2:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(target_q_finite, realized_quantiles)
            m[f"{strategy_name}/target_vs_realized_rank_corr"] = float(corr)
        else:
            m[f"{strategy_name}/target_vs_realized_rank_corr"] = math.nan

        top_threshold = max(0, n_quantiles - max(1, n_quantiles // 5))
        m[f"{strategy_name}/top_quantile_hit_rate"] = float(np.mean(realized_quantiles >= top_threshold))
        m[f"{strategy_name}/exact_quantile_accuracy"] = float(np.mean(target_q_finite == realized_quantiles))

        # per-bin calibration curve + monotonicity + MI
        m.update(conditioning_fidelity_metrics(strategy_name, target_q_finite, realized_quantiles, n_quantiles))

        # 4. Diversity — intra-bin pairwise distance
        m.update(intra_bin_diversity_metrics(f"{strategy_name}/states", states[finite], target_q_finite, n_quantiles))

        all_metrics.update(m)
        summary = {k: v for k, v in m.items() if math.isfinite(v) and "per_bin" not in k}
        print(json.dumps({strategy_name: summary}, sort_keys=True), flush=True)

    (output_dir / "score_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    # Print summary without per-bin detail
    print(json.dumps({k: v for k, v in all_metrics.items() if "per_bin" not in k}, sort_keys=True), flush=True)
    return all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Sweep over architectures and hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

# Each entry: fields understood by train_loss_generator, plus "name".
# ── ODE-steps axis (h=128, rb=2, nq=10): main hypothesis — more steps → lower TV
# ── Capacity axis (s=64): does larger h help when steps are right?
# ── Residual-blocks axis (s=64, h=128): do residuals help or hurt?
# ── n_quantiles axis (s=64, h=128, rb=2): granularity of conditioning signal
# ── quant_embed_dim axis (s=64, h=128, rb=2, nq=10): embedding size effect
SWEEP_CONFIGS = [
    # --- ODE steps ---
    {"name": "fm_s16",       "generator": "flow_matching", "hidden": 128, "steps": 16,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s32",       "generator": "flow_matching", "hidden": 128, "steps": 32,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s64",       "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s128",      "generator": "flow_matching", "hidden": 128, "steps": 128, "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    # --- Capacity (at s=64) ---
    {"name": "fm_s64_h64",   "generator": "flow_matching", "hidden": 64,  "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s64_h256",  "generator": "flow_matching", "hidden": 256, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    # --- Residual blocks (at s=64, h=128) ---
    {"name": "fm_s64_rb0",   "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 0, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s64_rb4",   "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 4, "kernel_size": 7, "n_quantiles": 10},
    # --- n_quantiles (at s=64, h=128, rb=2) ---
    {"name": "fm_s64_nq5",   "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 5},
    {"name": "fm_s64_nq20",  "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 20},
    # --- quantile embed dim (at s=64, h=128, rb=2, nq=10; default = max(8, 128//8) = 16) ---
    {"name": "fm_s64_qed8",  "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10, "quant_embed_dim": 8},
    {"name": "fm_s64_qed32", "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10, "quant_embed_dim": 32},
    {"name": "fm_s64_qed64", "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10, "quant_embed_dim": 64},
    # --- production-candidate confirmation: best granularity (nq) at the lean hidden=64 backbone ---
    {"name": "fm_s64_h64_nq20", "generator": "flow_matching", "hidden": 64, "steps": 64, "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 20},
    {"name": "fm_s64_h64_nq5",  "generator": "flow_matching", "hidden": 64, "steps": 64, "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 5},
]


def run_sweep(
    dataset_path: str | Path,
    output_dir: str | Path,
    configs: list[dict] | None = None,
    param_mode: str = "condition",
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 2e-4,
    n_quantiles: int = 10,
    seed: int = 0,
    device_name: str = "auto",
    n_score_samples: int = 1024,
    solver_batch_size: int = 512,
    jax_platform: str = "auto",
    skip_existing: bool = True,
    eval_every: int = 10,
    eval_n_samples: int = 512,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
) -> dict[str, dict]:
    if configs is None:
        configs = SWEEP_CONFIGS
    output_dir = Path(output_dir)
    summary: dict[str, dict] = {}
    for cfg in configs:
        name = cfg["name"]
        run_dir = output_dir / name
        # Resume: skip if score_metrics.json already exists
        if skip_existing and (run_dir / "score_metrics.json").exists():
            print(f"\nSkipping {name} (score_metrics.json exists)", flush=True)
            with open(run_dir / "score_metrics.json") as f:
                score = json.load(f)
            with open(run_dir / "history.json") as f:
                h = json.load(f)
            summary[name] = {**h[-1], **{k: v for k, v in score.items() if "top1/" in k or "uniform/" in k or "model/" in k or "reference/" in k}}
            continue
        print(f"\n{'='*60}\nSweep run: {name}\n{'='*60}", flush=True)
        t0 = time.perf_counter()
        # n_quantiles and quant_embed_dim can be overridden per-config entry
        cfg_nq = int(cfg.get("n_quantiles", n_quantiles))
        cfg_qed = int(cfg.get("quant_embed_dim", 0))
        ckpt_path, last_loss = train_loss_generator(
            dataset_path=dataset_path,
            output_dir=run_dir,
            generator=cfg.get("generator", "flow_matching"),
            param_mode=param_mode,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden=cfg.get("hidden", 128),
            steps=cfg.get("steps", 16),
            residual_blocks=cfg.get("residual_blocks", 2),
            kernel_size=cfg.get("kernel_size", 7),
            n_quantiles=cfg_nq,
            quant_embed_dim=cfg_qed,
            seed=seed,
            device_name=device_name,
            eval_every=eval_every,
            eval_n_samples=eval_n_samples,
            eval_solver_batch_size=solver_batch_size,
            eval_jax_platform=jax_platform,
            wandb_project=wandb_project,
            wandb_group=wandb_group or output_dir.name,
            wandb_run_name=name,
            wandb_extra_config={"sweep_name": output_dir.name, "config_name": name, **{k: cfg[k] for k in cfg if k != "name"}},
        )
        train_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        score = score_generated_samples(
            checkpoint_path=ckpt_path,
            dataset_path=dataset_path,
            output_dir=run_dir,
            n_samples=n_score_samples,
            seed=seed,
            device_name=device_name,
            solver_batch_size=solver_batch_size,
            jax_platform=jax_platform,
        )
        score_time = time.perf_counter() - t1
        summary[name] = {
            **last_loss,
            "train_time_s": train_time,
            "score_time_s": score_time,
            **{k: v for k, v in score.items() if "top1/" in k or "uniform/" in k or "model/" in k or "reference/" in k},
        }
        print(json.dumps({"run": name, **summary[name]}, sort_keys=True), flush=True)

    (output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    _print_sweep_table(summary, n_quantiles)
    return summary


def _print_sweep_table(summary: dict[str, dict], n_quantiles: int) -> None:
    """Print comparison table across all five evaluation axes."""
    W = 158
    print("\n" + "=" * W)
    hdr = (
        f"{'Model':<22}"
        f" {'params':>7} {'s':>4} {'nq':>3}"
        f" {'TV':>6} {'TV_r':>5}"          # realism: TV ratio
        f" {'amp_r':>6} {'psd_l2':>7}"     # realism: amplitude ratio, PSD distance
        f" {'top1_r':>7} {'uni_r':>6}"     # hardness: loss ratios
        f" {'cal_mae':>7} {'mono':>6} {'MI':>5}"  # conditioning fidelity
        f" {'ibd':>6}"                     # intra-bin diversity
        f" {'best_ep':>7} {'conv_ev':>7} {'conv_tr':>7}"  # convergence
        f" {'epoch_s':>7} {'ms/s':>6}"     # speed
    )
    print(hdr)
    print("-" * W)
    ref_tv = next(iter(summary.values())).get("reference_states/total_variation_mean", float("nan")) if summary else float("nan")
    val_tv = next(iter(summary.values())).get("validation_states/total_variation_mean", float("nan")) if summary else float("nan")
    print(f"  Reference (pool, uniform): TV={ref_tv:.1f}    Validation (real KS): TV={val_tv:.1f}")
    print("-" * W)
    for name, m in summary.items():
        tv = m.get("top1/states/total_variation_mean", float("nan"))
        tv_ref = val_tv if math.isfinite(val_tv) else ref_tv
        tv_ratio = tv / tv_ref if math.isfinite(tv) and tv_ref > 0 else float("nan")
        row = (
            f"{name:<22}"
            f" {m.get('model/n_params', 0):>7.0f}"
            f" {m.get('model/steps', 0):>4.0f}"
            f" {m.get('model/n_quantiles', 0):>3.0f}"
            # realism
            f" {tv:>6.1f}"
            f" {tv_ratio:>5.2f}"
            f" {m.get('top1/states/amplitude_ratio', float('nan')):>6.2f}"
            f" {m.get('top1/states/psd_log_l2', float('nan')):>7.2f}"
            # hardness
            f" {m.get('top1/loss_vs_reference_ratio', float('nan')):>7.2f}"
            f" {m.get('uniform/loss_vs_reference_ratio', float('nan')):>6.2f}"
            # conditioning fidelity (on top1 strategy)
            f" {m.get('top1/calibration_mae', float('nan')):>7.3f}"
            f" {m.get('top1/calibration_monotone_frac', float('nan')):>6.2f}"
            f" {m.get('top1/conditioning_mi_nats', float('nan')):>5.2f}"
            # diversity
            f" {m.get('top1/states/intra_bin_diversity_mean', float('nan')):>6.3f}"
            # convergence
            f" {m.get('best_eval_epoch', float('nan')):>7.0f}"
            f" {m.get('convergence_epoch_eval', float('nan')):>7.0f}"
            f" {m.get('convergence_epoch', float('nan')):>7.0f}"
            # speed
            f" {m.get('epoch_time_s', float('nan')):>7.2f}"
            f" {m.get('top1/gen_time_per_sample_ms', float('nan')):>6.2f}"
        )
        print(row)
    print("=" * W)
    print("Columns: TV=total-variation(top1), TV_r=TV/val_TV (1=perfect), amp_r=amplitude ratio (1=perfect),")
    print("         psd_l2=log-PSD L2 vs val (0=perfect), top1_r/uni_r=loss ratio vs reference (>1=harder),")
    print("         cal_mae=conditioning calibration err (0=perfect), mono=monotone frac (1=perfect),")
    print("         MI=mutual info nats (log(nq)=perfect), ibd=intra-bin diversity,")
    print("         best_ep=eval epoch with best combined score, conv_ev=first eval epoch within 5% of best,")
    print("         conv_tr=convergence epoch on train loss, epoch_s=mean s/epoch, ms/s=gen ms/sample")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry points
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--n-quantile-bins", type=int, default=10)
    parser.add_argument("--recompute-with-final-surrogate", action="store_true",
                        help="Re-evaluate pool losses using surrogate.pt instead of the per-round losses")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    path = build_loss_dataset_from_run(
        args.run_dir,
        output_path=args.output,
        round_id=args.round,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        n_quantile_bins=args.n_quantile_bins,
        recompute_with_final_surrogate=args.recompute_with_final_surrogate,
        device_name=args.device,
    )
    print(path)


def train_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generator", choices=["ddpm", "flow_matching"], default="flow_matching")
    parser.add_argument("--param-mode", choices=["condition", "generate"], default="condition")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--loss-metric", choices=["mse", "rmse"], default="rmse")
    parser.add_argument("--n-quantiles", type=int, default=10)
    parser.add_argument("--quant-embed-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jax-platform", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--score-samples", type=int, default=1024)
    parser.add_argument("--solver-batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run full 5-axis validation eval every N epochs (0 = disable)")
    parser.add_argument("--eval-n-samples", type=int, default=512)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args(argv)
    ckpt, _ = train_loss_generator(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        generator=args.generator,
        param_mode=args.param_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        steps=args.steps,
        residual_blocks=args.residual_blocks,
        kernel_size=args.kernel_size,
        loss_metric=args.loss_metric,
        n_quantiles=args.n_quantiles,
        quant_embed_dim=args.quant_embed_dim,
        seed=args.seed,
        device_name=args.device,
        eval_every=args.eval_every,
        eval_n_samples=args.eval_n_samples,
        eval_solver_batch_size=args.solver_batch_size,
        eval_jax_platform=args.jax_platform,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
    )
    if args.score_samples > 0:
        score_generated_samples(
            ckpt,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            n_samples=args.score_samples,
            seed=args.seed,
            device_name=args.device,
            solver_batch_size=args.solver_batch_size,
            jax_platform=args.jax_platform,
        )


def sweep_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--param-mode", choices=["condition", "generate"], default="condition")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-quantiles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-samples", type=int, default=1024)
    parser.add_argument("--solver-batch-size", type=int, default=512)
    parser.add_argument("--jax-platform", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Run full 5-axis validation eval every N epochs (0 = disable)")
    parser.add_argument("--eval-n-samples", type=int, default=512,
                        help="Samples per strategy at intermediate evals (final score uses --score-samples)")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project for per-config runs (omit to disable)")
    parser.add_argument("--wandb-group", default=None,
                        help="W&B group (default: sweep output dir name)")
    args = parser.parse_args(argv)
    run_sweep(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        param_mode=args.param_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_quantiles=args.n_quantiles,
        seed=args.seed,
        device_name=args.device,
        n_score_samples=args.score_samples,
        solver_batch_size=args.solver_batch_size,
        eval_every=args.eval_every,
        eval_n_samples=args.eval_n_samples,
        jax_platform=args.jax_platform,
        skip_existing=not args.no_skip_existing,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
    )
