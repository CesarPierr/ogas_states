"""End-of-round evaluation of the production generator (flow / ddpm).

Computes the 10 most informative generator-health metrics each round and returns
them under the ``generator_metrics/`` namespace for W&B. Metric formulas are the
same as the ablation lab (loss_generator_lab) so production numbers are directly
comparable to the offline architecture sweep.

The 10 metrics, by axis:
  Realism   : tv_ratio, psd_log_l2, amplitude_ratio        (vs real-KS validation states)
  Validity  : finite_ratio
  Hardness  : hardness_ratio                                 (surrogate loss vs current pool)
  Conditioning : rank_corr, calibration_mae, conditioning_mi_norm
  Diversity : intra_bin_diversity, pairwise_l2_mean
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .data import TransitionPool
from .loss_generator_lab import (
    amplitude_distribution_metrics,
    conditioning_fidelity_metrics,
    intra_bin_diversity_metrics,
    pairwise_l2_stats,
    psd_similarity_metrics,
    state_quality_metrics,
)
from .train import (
    compute_transition_losses,
    compute_transition_uncertainty,
    normalize_losses,
    realized_difficulty,
    sample_quantile_labels,
)


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


@torch.no_grad()
def evaluate_generator(
    ddpm,
    surrogate,
    pde,
    val_ref_states: np.ndarray | None,
    ref_loss_mean: float,
    state_mean: float,
    state_std: float,
    rng: np.random.Generator,
    device: torch.device,
    loss_metric: str = "rmse",
    difficulty_signal: str = "loss",
    n_eval: int = 512,
    solver_batch_size: int = 512,
    surrogate_batch_size: int = 512,
) -> dict[str, float]:
    """Generate a probe batch (bins drawn uniformly so conditioning is measurable),
    step it through the solver, score with the surrogate, and return the 10 metrics.

    Requires ddpm.n_quantiles > 0 and ddpm.quantile_edges set (by train_ddpm).
    """
    n_quantiles = int(getattr(ddpm, "n_quantiles", 0))
    quantile_edges = getattr(ddpm, "quantile_edges", None)
    if n_quantiles <= 0 or quantile_edges is None:
        return {}
    quantile_edges = np.asarray(quantile_edges, dtype=np.float32)

    # 1. Probe targets: uniform across all bins so target/realized correlation is identifiable.
    target_bins = sample_quantile_labels("uniform", n_eval, n_quantiles, rng)
    # 2. Params drawn uniformly (the prod conditioning prior) and conditioned on.
    params_phys = pde.sample_params_uniform(n_eval, rng)
    param_scaled = torch.from_numpy(2.0 * pde.normalize_params(params_phys) - 1.0).to(device)
    qlabel = torch.as_tensor(target_bins, dtype=torch.long, device=device)
    ddpm.eval()
    state_norm, gen_p = ddpm.sample(
        n_eval,
        loss=None,
        params=param_scaled if ddpm.use_param_cond else None,
        device=device,
        temperature=1.0,
        quantile_label=qlabel,
    )
    states = (state_norm.cpu().numpy() * state_std + state_mean).astype(np.float32)
    if ddpm.generate_params and gen_p is not None:
        normed = ((gen_p.clamp(-1.0, 1.0).cpu().numpy() + 1.0) / 2.0).astype(np.float32)
        params_phys = pde.denormalize_params(normed)

    # 3. Step through the real solver and score with the surrogate.
    next_states = _pde_step_batch(pde, states, params_phys, solver_batch_size)
    pool = TransitionPool(states=states, params=params_phys, next_states=next_states)
    gen_losses = compute_transition_losses(surrogate, pool, batch_size=surrogate_batch_size, device=device)
    gen_uncertainty = (
        compute_transition_uncertainty(surrogate, pool, batch_size=surrogate_batch_size, device=device)
        if getattr(surrogate, "n_models", 1) > 1
        else None
    )
    finite = (
        np.isfinite(states).all(axis=(1, 2))
        & np.isfinite(next_states).all(axis=(1, 2))
        & np.isfinite(gen_losses)
    )

    out: dict[str, float] = {"generator_metrics/finite_ratio": float(finite.mean())}
    if not finite.any():
        return out

    sf = states[finite]
    gl = gen_losses[finite]
    tb = target_bins[finite]

    # --- Realism (vs real-KS validation states) ---
    sq = state_quality_metrics("g", sf)
    gen_tv = sq.get("g/total_variation_mean", float("nan"))
    if val_ref_states is not None and len(val_ref_states) > 1:
        ref_tv = float(np.mean([np.sum(np.abs(np.diff(v.reshape(-1)))) for v in val_ref_states]))
        out["generator_metrics/tv_ratio"] = gen_tv / ref_tv if ref_tv > 0 else float("nan")
        psd = psd_similarity_metrics("g", sf, val_ref_states)
        out["generator_metrics/psd_log_l2"] = psd.get("g/psd_log_l2", float("nan"))
        amp = amplitude_distribution_metrics("g", sf, val_ref_states)
        out["generator_metrics/amplitude_ratio"] = amp.get("g/amplitude_ratio", float("nan"))
    else:
        out["generator_metrics/tv_ratio"] = float("nan")
        out["generator_metrics/psd_log_l2"] = float("nan")
        out["generator_metrics/amplitude_ratio"] = float("nan")

    # --- Hardness (vs current pool difficulty) ---
    out["generator_metrics/hardness_ratio"] = float(np.mean(gl)) / (float(ref_loss_mean) + 1e-12)

    # --- Conditioning (realized bin from surrogate loss vs target bin) ---
    # Use the same difficulty signal as training so realized bins align with the edges.
    gu = gen_uncertainty[finite] if gen_uncertainty is not None else None
    realized_norm = normalize_losses(
        realized_difficulty(gl, gu, next_states[finite], loss_metric, difficulty_signal)
    )
    realized_bins = np.digitize(realized_norm, quantile_edges[1:], right=False).clip(0, n_quantiles - 1)
    if len(realized_bins) > 2:
        from scipy.stats import spearmanr

        corr, _ = spearmanr(tb, realized_bins)
        out["generator_metrics/rank_corr"] = float(corr)
    else:
        out["generator_metrics/rank_corr"] = float("nan")
    cond = conditioning_fidelity_metrics("g", tb, realized_bins, n_quantiles)
    out["generator_metrics/calibration_mae"] = cond.get("g/calibration_mae", float("nan"))
    mi = cond.get("g/conditioning_mi_nats", float("nan"))
    out["generator_metrics/conditioning_mi_norm"] = (
        mi / math.log(n_quantiles) if math.isfinite(mi) and n_quantiles > 1 else float("nan")
    )

    # --- Diversity ---
    div = intra_bin_diversity_metrics("g", sf, tb, n_quantiles)
    out["generator_metrics/intra_bin_diversity"] = div.get("g/intra_bin_diversity_mean", float("nan"))
    pl = pairwise_l2_stats(sf)
    out["generator_metrics/pairwise_l2_mean"] = pl.get("pairwise_l2_mean", float("nan"))

    return out
