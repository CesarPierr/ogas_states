"""Weights & Biases setup and per-round diagnostics logging.

``wandb`` is imported lazily inside the functions so the package imports without it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import TransitionPool
from .eval import (
    evaluate_one_step_validation,
    rollout_error_series,
    rollout_spacetime_examples,
)
from .viz import rollout_error_pyplot_image, spacetime_grid_image


def wandb_group_label(cfg) -> str:
    label = str(cfg.wandb.group or "default")
    for prefix in ("ks800_5case_", "ks800_al4pde_exact_5seed_viz_", "ks800_ready_"):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def epoch_validation_count(cfg, validation) -> int:
    ten_percent = max(1, int(np.ceil(0.10 * validation.n_trajectories)))
    configured = int(getattr(cfg.validation, "epoch_n_trajectories", ten_percent) or ten_percent)
    return max(1, min(configured, ten_percent, validation.n_trajectories))


def maybe_wandb(cfg, run_id: str | None):
    if not cfg.wandb.enabled:
        return None
    import wandb

    # Generate a hash of the config (excluding seed, device, output_dir, wandb settings)
    # so that experiments with the same config but different seeds share the same group.
    import hashlib
    import json
    cfg_dict = asdict(cfg)
    cfg_dict.pop("seed", None)
    cfg_dict.pop("device", None)
    cfg_dict.pop("output_dir", None)
    cfg_dict.pop("wandb", None)
    
    def json_serializable(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (tuple, set)):
            return list(obj)
        return str(obj)

    cfg_json = json.dumps(cfg_dict, sort_keys=True, default=json_serializable)
    cfg_hash = hashlib.md5(cfg_json.encode("utf-8")).hexdigest()[:8]

    base_group = cfg.wandb.group or "default"
    if cfg_hash not in base_group:
        group_name = f"{base_group}_{cfg_hash}"
    else:
        group_name = base_group

    kwargs = {}
    if not os.environ.get("WANDB_API_KEY"):
        kwargs["anonymous"] = "allow"
    run = wandb.init(
        project=cfg.wandb.project,
        group=group_name,
        job_type=wandb_group_label(cfg),
        tags=[wandb_group_label(cfg), f"seed-{cfg.seed}"],
        config=asdict(cfg),
        id=run_id,
        name=wandb_group_label(cfg),
        resume="allow" if run_id else None,
        **kwargs,
    )
    # Define semantic axes so runs with and without DDPM logs stay comparable.
    run.define_metric("surrogate_epoch/global_epoch")
    run.define_metric("ddpm_epoch/global_epoch")
    run.define_metric("round")

    run.define_metric("surrogate_epoch/*", step_metric="surrogate_epoch/global_epoch")
    run.define_metric("surrogate_epoch_val/*", step_metric="surrogate_epoch/global_epoch")
    run.define_metric("ddpm_epoch/*", step_metric="ddpm_epoch/global_epoch")

    for namespace in (
        "val",
        "train",
        "ddpm",
        "pool",
        "rollout",
        "pool_dist",
        "rollout_figures",
        "generator_metrics",
        "timing",
        "hard_val",
    ):
        run.define_metric(f"{namespace}/*", step_metric="round")

    return run


def log_wandb_diagnostics(
    run,
    model,
    ddpm,
    pool: TransitionPool,
    validation,
    cfg,
    device,
    round_id: int,
    step: int,
    pde: PDE1D,
    state_mean: float,
    state_std: float,
) -> None:
    import wandb

    payload = {}
    if pool.losses is not None:
        payload["pool_dist/loss_hist"] = wandb.Histogram(pool.losses)
    if pool.source is not None:
        source = pool.source.reshape(-1)
        for value, name in ((0, "uniform"), (1, "generated")):
            mask = source == value
            if not mask.any():
                continue
            if pool.losses is not None:
                payload[f"pool_dist/{name}_posttrain_loss_hist"] = wandb.Histogram(pool.losses[mask])
            if pool.pretrain_losses is not None:
                payload[f"pool_dist/{name}_pretrain_loss_hist"] = wandb.Histogram(pool.pretrain_losses[mask])
            for j in range(pool.params.shape[1]):
                payload[f"pool_dist/{name}_param{j}_hist"] = wandb.Histogram(pool.params[mask, j])

    steps = min(cfg.validation.rollout_steps, validation.trajectory_steps)
    spacetime = rollout_spacetime_examples(model, validation, steps=steps, device=device, n_examples=3)
    payload["rollout_figures/spacetime_grid"] = wandb.Image(
        spacetime_grid_image(spacetime, n_rows=3),
        caption=(
            "3 fixed validation rollouts (rows, evenly spaced indices) x "
            "[ground truth | prediction | absolute error]. Same examples every round/run. "
            "Per-row color scale anchored to that example's ground-truth q98 amplitude."
        ),
    )

    series = rollout_error_series(model, validation, steps=steps, device=device, n_trajectories=256)
    payload["rollout_figures/rmse_per_step"] = wandb.Image(
        rollout_error_pyplot_image(series, "rmse"),
        caption="Rollout RMSE mean/p95/p99 per forecast step",
    )
    payload["rollout_figures/nrmse_per_step"] = wandb.Image(
        rollout_error_pyplot_image(series, "nrmse"),
        caption="Rollout NRMSE mean/p95/p99 per forecast step",
    )
    hard_spacetime = rollout_spacetime_examples(
        model, validation, steps=steps, device=device, n_examples=3, hard=True
    )
    payload["rollout_figures/spacetime_hard_grid"] = wandb.Image(
        spacetime_grid_image(hard_spacetime, n_rows=3),
        caption=(
            "3 hardest validation rollouts (rows, ranked by ground-truth spatial total-variation, "
            "model-independent so fixed across rounds/runs) x [ground truth | prediction | absolute error]. "
            "Per-row color scale anchored to that example's ground-truth q98 amplitude."
        ),
    )
    payload["round"] = round_id
    run.log(payload)


def make_epoch_callback(run, model, validation, cfg, device, round_id: int, prefix: str):
    if run is None:
        return None

    def _callback(metrics: dict[str, float], epoch: int) -> None:
        epoch_metrics = {f"{prefix}/{k.split('/', 1)[-1]}": v for k, v in metrics.items()}
        epoch_metrics["round"] = round_id
        epoch_metrics["epoch"] = epoch
        if prefix == "surrogate_epoch":
            epoch_n_trajectories = epoch_validation_count(cfg, validation)
            val = evaluate_one_step_validation(
                model,
                validation,
                cfg.validation.quantiles,
                device,
                n_trajectories=epoch_n_trajectories,
                prefix="surrogate_epoch_val",
            )
            val["surrogate_epoch_val/n_trajectories"] = float(epoch_n_trajectories)
            val["surrogate_epoch_val/fraction"] = float(
                epoch_n_trajectories / max(1, validation.n_trajectories)
            )
            epoch_metrics.update(val)
        run.log(epoch_metrics)

    return _callback
