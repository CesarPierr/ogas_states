"""Pool/state diagnostic metrics. Pure functions returning plain dicts (no backend)."""
from __future__ import annotations

import numpy as np

from .data import TransitionPool
from .train import normalize_losses, transform_transition_losses


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def state_diversity(states: np.ndarray, max_points: int = 512) -> dict[str, float]:
    if len(states) < 2:
        return {"pairwise_l2_mean": 0.0, "pairwise_l2_p10": 0.0, "pairwise_l2_p90": 0.0}
    idx = np.linspace(0, len(states) - 1, min(max_points, len(states)), dtype=np.int64)
    flat = states[idx].reshape(len(idx), -1).astype(np.float32)
    norms = np.sum(flat**2, axis=1, keepdims=True)
    dist2 = np.maximum(norms + norms.T - 2.0 * flat @ flat.T, 0.0)
    tri = np.sqrt(dist2[np.triu_indices(len(idx), k=1)])
    return {
        "pairwise_l2_mean": float(tri.mean()),
        "pairwise_l2_p10": float(np.quantile(tri, 0.1)),
        "pairwise_l2_p90": float(np.quantile(tri, 0.9)),
    }


def metric_name(prefix: str, suffix: str) -> str:
    if "/" in prefix:
        namespace, base = prefix.split("/", 1)
        return f"{namespace}/{base}_{suffix}"
    return f"{prefix}/{suffix}"


def state_quality_metrics(states: np.ndarray, prefix: str) -> dict[str, float]:
    flat = np.asarray(states, dtype=np.float32)
    finite = np.isfinite(flat)
    metrics = {metric_name(prefix, "finite_ratio"): float(finite.mean())}
    if not finite.any():
        return metrics
    clean = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    dx = np.diff(clean, axis=-1, append=clean[..., :1])
    spectrum = np.fft.rfft(clean.reshape(-1, clean.shape[-1]), axis=-1)
    power = np.abs(spectrum) ** 2
    cutoff = max(2, power.shape[-1] // 3)
    total_power = power.sum(axis=-1) + 1e-8
    high_power = power[:, cutoff:].sum(axis=-1)
    metrics.update(
        {
            metric_name(prefix, "value_std"): float(clean.std()),
            metric_name(prefix, "abs_p99"): float(np.quantile(np.abs(clean), 0.99)),
            metric_name(prefix, "total_variation_mean"): float(np.mean(np.sum(np.abs(dx), axis=-1))),
            metric_name(prefix, "roughness_mean"): float(np.mean(dx**2)),
            metric_name(prefix, "highfreq_energy_ratio_mean"): float(np.mean(high_power / total_power)),
        }
    )
    return metrics


def pool_debug_metrics(
    pool: TransitionPool,
    pretrain_losses: np.ndarray | None = None,
    loss_metric: str = "mse",
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if pool.source is None:
        return metrics
    source = pool.source.reshape(-1)
    names = {0: "uniform", 1: "generated"}
    for value, name in names.items():
        mask = source == value
        metrics[f"pool/{name}_n"] = float(mask.sum())
        if not mask.any():
            continue
        states = pool.states[mask]
        if name == "generated":
            div = state_diversity(states)
            metrics[f"pool/{name}_pairwise_l2_mean"] = div["pairwise_l2_mean"]
            metrics[f"pool/{name}_pairwise_l2_p10"] = div["pairwise_l2_p10"]
            metrics[f"pool/{name}_pairwise_l2_p90"] = div["pairwise_l2_p90"]
            metrics.update(state_quality_metrics(states, f"pool/{name}_states"))
        if pretrain_losses is not None:
            losses = pretrain_losses[mask]
            metrics[f"pool/{name}_pretrain_loss_mean"] = float(losses.mean())
            metrics[f"pool/{name}_pretrain_loss_p90"] = float(np.quantile(losses, 0.9))
            if name == "generated":
                metrics[f"pool/{name}_pretrain_loss_p99"] = float(np.quantile(losses, 0.99))
        if pool.losses is not None:
            losses = pool.losses[mask]
            metrics[f"pool/{name}_posttrain_loss_mean"] = float(losses.mean())
            metrics[f"pool/{name}_posttrain_loss_p90"] = float(np.quantile(losses, 0.9))
            if name == "generated":
                metrics[f"pool/{name}_posttrain_loss_p99"] = float(np.quantile(losses, 0.99))
    if (
        pretrain_losses is not None
        and pool.proposed_losses is not None
        and np.isfinite(pool.proposed_losses).any()
    ):
        gen = source == 1
        proposed = pool.proposed_losses[gen]
        proposed = proposed[np.isfinite(proposed)]
        if len(proposed) > 0:
            metrics["pool/generated_proposed_loss_mean"] = float(proposed.mean())
            metrics["pool/generated_proposed_loss_p90"] = float(np.quantile(proposed, 0.9))
            metrics["pool/generated_proposed_loss_p99"] = float(np.quantile(proposed, 0.99))
        pretrain_condition_losses = normalize_losses(
            transform_transition_losses(pretrain_losses[gen], loss_metric)
        )
        metrics["pool/generated_proposed_vs_pretrain_loss_corr"] = safe_corr(
            pool.proposed_losses[gen], pretrain_condition_losses
        )
    if pool.losses is not None and pool.source is not None:
        gen = source == 1
        if gen.any() and pool.proposed_losses is not None:
            posttrain_condition_losses = normalize_losses(
                transform_transition_losses(pool.losses[gen], loss_metric)
            )
            metrics["pool/generated_proposed_vs_posttrain_loss_corr"] = safe_corr(
                pool.proposed_losses[gen], posttrain_condition_losses
            )
    return metrics
