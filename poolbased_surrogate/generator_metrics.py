"""Generator realism / conditioning-fidelity metrics.

Shared by the live per-round generator evaluation (``generator_eval``) and the
standalone loss-generator lab. Pure functions over numpy arrays; scipy is imported
lazily inside the one function that needs it.
"""
from __future__ import annotations

import math

import numpy as np


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
