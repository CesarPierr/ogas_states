"""Score generated samples against real-KS reference statistics."""
from __future__ import annotations

import json
import math
import numpy as np
import os
import time
import torch
from pathlib import Path
from .data import TransitionPool
from .generator_metrics import amplitude_distribution_metrics, conditioning_fidelity_metrics, intra_bin_diversity_metrics, psd_similarity_metrics, state_quality_metrics
from .models.surrogate import build_surrogate
from .pde import PDE1D
from .train import compute_transition_losses, normalize_losses, transform_transition_losses
from ._lg_common import assign_loss_quantile_bins, choose_device, config_from_resolved_dict
from ._lg_generator import _sampling_strategies, sample_loss_generator


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
