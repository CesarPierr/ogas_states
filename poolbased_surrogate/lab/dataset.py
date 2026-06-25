"""Build the (state, loss-quantile) dataset from a finished training run."""
from __future__ import annotations

import json
import numpy as np
import time
import torch
from pathlib import Path
from ..data import TransitionPool
from ..models.surrogate import build_surrogate
from ..pde import PDE1D
from ..train import compute_transition_losses, normalize_losses, transform_transition_losses
from .common import assign_loss_quantile_bins, choose_device, load_resolved_config, loss_norm_constants, stratified_split


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
    from ..models.surrogate import build_surrogate
    from ..data import TransitionPool
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
    # Training-pool normalization constants (q05/q95 per loss metric), persisted so
    # scoring can renormalize generated losses on the TRAINING scale (voir scoring.py).
    loss_norm_lo_mse, loss_norm_hi_mse = loss_norm_constants(losses)
    loss_norm_lo_rmse, loss_norm_hi_rmse = loss_norm_constants(transform_transition_losses(losses, "rmse"))
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
        "loss_norm_lo_mse": loss_norm_lo_mse,
        "loss_norm_hi_mse": loss_norm_hi_mse,
        "loss_norm_lo_rmse": loss_norm_lo_rmse,
        "loss_norm_hi_rmse": loss_norm_hi_rmse,
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
        loss_norm_lo_mse=np.array(loss_norm_lo_mse, dtype=np.float32),
        loss_norm_hi_mse=np.array(loss_norm_hi_mse, dtype=np.float32),
        loss_norm_lo_rmse=np.array(loss_norm_lo_rmse, dtype=np.float32),
        loss_norm_hi_rmse=np.array(loss_norm_hi_rmse, dtype=np.float32),
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
