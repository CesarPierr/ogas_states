from __future__ import annotations

import json
import os
import random
import sys
import argparse
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import load_config
from .data import TransitionPool
from .eval import (
    create_validation_data,
    evaluate_one_step_validation,
    evaluate_validation,
    load_validation_data,
    one_step_examples,
    rollout_error_series,
    rollout_spacetime_examples,
    save_validation_data,
)
from .models.ddpm import DDPM1D
from .models.surrogate import build_surrogate
from .pde import PDE1D
from .train import compute_transition_losses, propose_losses, train_ddpm, train_surrogate


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_uniform_pool(pde: PDE1D, n_traj: int, steps: int, rng: np.random.Generator) -> TransitionPool:
    params = pde.sample_params_uniform(n_traj, rng)
    states0 = pde.sample_ic_uniform(n_traj, rng)
    traj = pde.simulate(states0, params, steps)
    return TransitionPool.from_trajectories(traj, params)


@torch.no_grad()
def generate_states_and_params(
    ddpm: DDPM1D,
    n: int,
    mode: str,
    loss_values: np.ndarray | None,
    pde: PDE1D,
    state_mean: float,
    state_std: float,
    rng: np.random.Generator,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate states (physical units) and their PDE parameters from the DDPM.

    Depending on ddpm.param_mode the parameters are either sampled uniformly and used to
    condition the state generation, jointly generated with the state, or sampled uniformly
    and paired afterwards ("none"). The returned params are exactly the ones to feed the
    solver and the surrogate.
    """
    states_out, params_out = [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = end - start
        loss_cond = None
        if mode == "conditional_loss" and loss_values is not None:
            loss_cond = torch.as_tensor(loss_values[start:end], dtype=torch.float32, device=device).view(b, 1)
        if ddpm.param_mode == "condition":
            normed = rng.random((b, ddpm.param_dim)).astype(np.float32)
            param_scaled = torch.from_numpy(2.0 * normed - 1.0).to(device)
            state_norm, _ = ddpm.sample(b, loss=loss_cond, params=param_scaled, device=device)
            params_phys = pde.denormalize_params(normed)
        elif ddpm.param_mode == "generate":
            state_norm, p_scaled = ddpm.sample(b, loss=loss_cond, params=None, device=device)
            normed = ((p_scaled.clamp(-1.0, 1.0).cpu().numpy() + 1.0) / 2.0).astype(np.float32)
            params_phys = pde.denormalize_params(normed)
        else:
            state_norm, _ = ddpm.sample(b, loss=loss_cond, params=None, device=device)
            params_phys = pde.sample_params_uniform(b, rng)
        state_phys = state_norm.cpu().numpy() * state_std + state_mean
        states_out.append(state_phys.astype(np.float32))
        params_out.append(params_phys.astype(np.float32))
    return np.concatenate(states_out, axis=0), np.concatenate(params_out, axis=0)


@torch.no_grad()
def make_mixed_pool(
    pde: PDE1D,
    ddpm: DDPM1D,
    previous: TransitionPool,
    n_traj: int,
    steps: int,
    uniform_fraction: float,
    ddpm_mode: str,
    rng: np.random.Generator,
    device: torch.device,
    state_mean: float,
    state_std: float,
) -> TransitionPool:
    n_uniform = int(round(n_traj * uniform_fraction))
    n_generated_traj = n_traj - n_uniform
    pools = []
    if n_uniform > 0:
        pools.append(make_uniform_pool(pde, n_uniform, steps, rng))
    if n_generated_traj > 0:
        n_generated = n_generated_traj * steps
        if previous.losses is None:
            loss_values = rng.uniform(0.5, 1.0, size=(n_generated, 1)).astype(np.float32)
        else:
            loss_values = propose_losses(previous.losses, n_generated, rng)
        states0, params = generate_states_and_params(
            ddpm,
            n_generated,
            ddpm_mode,
            loss_values if ddpm_mode == "conditional_loss" else None,
            pde,
            state_mean,
            state_std,
            rng,
            device,
        )
        next_state = pde.step(states0[:, 0].astype(np.float64), params)[:, None, :].astype(np.float32)
        pools.append(
            TransitionPool(
                states0,
                params,
                next_state,
                source=np.ones((n_generated,), dtype=np.int8),
                proposed_losses=loss_values.reshape(-1).astype(np.float32),
            )
        )
    states = np.concatenate([p.states for p in pools], axis=0)
    params = np.concatenate([p.params for p in pools], axis=0)
    next_states = np.concatenate([p.next_states for p in pools], axis=0)
    source = np.concatenate([p.source for p in pools if p.source is not None], axis=0)
    proposed = np.full((len(states),), np.nan, dtype=np.float32)
    offset = 0
    for p in pools:
        if p.proposed_losses is not None:
            proposed[offset : offset + len(p)] = p.proposed_losses.reshape(-1)
        offset += len(p)
    return TransitionPool(states, params, next_states, source=source, proposed_losses=proposed)


def maybe_wandb(cfg, run_id: str | None):
    if not cfg.wandb.enabled:
        return None
    import wandb

    kwargs = {}
    if not os.environ.get("WANDB_API_KEY"):
        kwargs["anonymous"] = "allow"
    return wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        config=asdict(cfg),
        id=run_id,
        name=Path(cfg.output_dir).name,
        resume="allow" if run_id else None,
        **kwargs,
    )


def save_checkpoint(
    path: Path,
    round_id: int,
    model,
    ddpm,
    pool: TransitionPool,
    history: list[dict[str, float]],
    rng: np.random.Generator,
    wandb_run_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_round": round_id + 1,
        "surrogate": model.state_dict(),
        "ddpm": ddpm.state_dict(),
        "pool": {
            "states": torch.from_numpy(pool.states),
            "params": torch.from_numpy(pool.params),
            "next_states": torch.from_numpy(pool.next_states),
            "losses": None if pool.losses is None else torch.from_numpy(pool.losses),
            "source": None if pool.source is None else torch.from_numpy(pool.source),
            "proposed_losses": None
            if pool.proposed_losses is None
            else torch.from_numpy(pool.proposed_losses),
            "pretrain_losses": None
            if pool.pretrain_losses is None
            else torch.from_numpy(pool.pretrain_losses),
        },
        "history": history,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "wandb_run_id": wandb_run_id,
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, ddpm, rng: np.random.Generator, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["surrogate"])
    ddpm.load_state_dict(payload["ddpm"])
    p = payload["pool"]
    pool = TransitionPool(
        states=p["states"].cpu().numpy(),
        params=p["params"].cpu().numpy(),
        next_states=p["next_states"].cpu().numpy(),
        losses=None if p["losses"] is None else p["losses"].cpu().numpy(),
        source=None if p.get("source") is None else p["source"].cpu().numpy(),
        proposed_losses=None
        if p.get("proposed_losses") is None
        else p["proposed_losses"].cpu().numpy(),
        pretrain_losses=None
        if p.get("pretrain_losses") is None
        else p["pretrain_losses"].cpu().numpy(),
    )
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "torch_cuda_rng_state" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state"])
    return int(payload["next_round"]), pool, list(payload["history"]), str(payload["wandb_run_id"])


def apply_override(cfg, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use key=value syntax, got {expression!r}")
    key, raw_value = expression.split("=", 1)
    target = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown override key {key!r}")
        target = getattr(target, part)
    final = parts[-1]
    if not hasattr(target, final):
        raise ValueError(f"Unknown override key {key!r}")
    value = yaml.safe_load(raw_value)
    current = getattr(target, final)
    if isinstance(current, tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(target, final, value)


def validation_path(cfg, out: Path) -> Path:
    if cfg.validation.path:
        return Path(cfg.validation.path)
    return out / "validation.npz"


def load_or_create_validation(cfg, pde: PDE1D, out: Path):
    path = validation_path(cfg, out)
    if path.exists():
        validation = load_validation_data(path)
        if (
            validation.n_trajectories == cfg.validation.n_trajectories
            and validation.trajectory_steps == cfg.validation.trajectory_steps
            and validation.states0.shape[-1] == cfg.pde.resolution
        ):
            print(f"loaded validation set from {path}")
            return validation
        raise ValueError(
            f"Validation set {path} does not match config: "
            f"file has {validation.n_trajectories} trajectories x "
            f"{validation.trajectory_steps} steps at resolution {validation.states0.shape[-1]}, "
            f"config asks for {cfg.validation.n_trajectories} trajectories x "
            f"{cfg.validation.trajectory_steps} steps at resolution {cfg.pde.resolution}."
        )
    validation = create_validation_data(
        pde=pde,
        n_trajectories=cfg.validation.n_trajectories,
        trajectory_steps=cfg.validation.trajectory_steps,
        seed=cfg.seed + 10_000,
    )
    save_validation_data(path, validation)
    print(
        "created validation set "
        f"{path} with {validation.n_trajectories} trajectories x {validation.trajectory_steps} steps"
    )
    return validation


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


def state_quality_metrics(states: np.ndarray, prefix: str) -> dict[str, float]:
    flat = np.asarray(states, dtype=np.float32)
    finite = np.isfinite(flat)
    metrics = {f"{prefix}/finite_ratio": float(finite.mean())}
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
            f"{prefix}/value_mean": float(clean.mean()),
            f"{prefix}/value_std": float(clean.std()),
            f"{prefix}/abs_p95": float(np.quantile(np.abs(clean), 0.95)),
            f"{prefix}/abs_p99": float(np.quantile(np.abs(clean), 0.99)),
            f"{prefix}/total_variation_mean": float(np.mean(np.sum(np.abs(dx), axis=-1))),
            f"{prefix}/roughness_mean": float(np.mean(dx**2)),
            f"{prefix}/highfreq_energy_ratio_mean": float(np.mean(high_power / total_power)),
            f"{prefix}/near_clip_ratio_5": float(np.mean(np.abs(clean) > 4.9)),
            f"{prefix}/near_clip_ratio_10": float(np.mean(np.abs(clean) > 9.8)),
        }
    )
    return metrics


def pool_debug_metrics(pool: TransitionPool, pretrain_losses: np.ndarray | None = None) -> dict[str, float]:
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
        params = pool.params[mask]
        div = state_diversity(states)
        for key, val in div.items():
            metrics[f"pool/{name}_{key}"] = val
        metrics.update(state_quality_metrics(states, f"pool/{name}_states"))
        for j in range(params.shape[1]):
            metrics[f"pool/{name}_param{j}_mean"] = float(params[:, j].mean())
            metrics[f"pool/{name}_param{j}_std"] = float(params[:, j].std())
            metrics[f"pool/{name}_param{j}_p05"] = float(np.quantile(params[:, j], 0.05))
            metrics[f"pool/{name}_param{j}_p95"] = float(np.quantile(params[:, j], 0.95))
        if pretrain_losses is not None:
            losses = pretrain_losses[mask]
            metrics[f"pool/{name}_pretrain_loss_mean"] = float(losses.mean())
            metrics[f"pool/{name}_pretrain_loss_p50"] = float(np.quantile(losses, 0.5))
            metrics[f"pool/{name}_pretrain_loss_p90"] = float(np.quantile(losses, 0.9))
            metrics[f"pool/{name}_pretrain_loss_p99"] = float(np.quantile(losses, 0.99))
        if pool.losses is not None:
            losses = pool.losses[mask]
            metrics[f"pool/{name}_posttrain_loss_mean"] = float(losses.mean())
            metrics[f"pool/{name}_posttrain_loss_p50"] = float(np.quantile(losses, 0.5))
            metrics[f"pool/{name}_posttrain_loss_p90"] = float(np.quantile(losses, 0.9))
            metrics[f"pool/{name}_posttrain_loss_p99"] = float(np.quantile(losses, 0.99))
    if (
        pretrain_losses is not None
        and pool.proposed_losses is not None
        and np.isfinite(pool.proposed_losses).any()
    ):
        gen = source == 1
        metrics["pool/generated_proposed_vs_pretrain_loss_corr"] = safe_corr(
            pool.proposed_losses[gen], pretrain_losses[gen]
        )
    if pool.losses is not None and pool.source is not None:
        gen = source == 1
        if gen.any() and pool.proposed_losses is not None:
            metrics["pool/generated_proposed_vs_posttrain_loss_corr"] = safe_corr(
                pool.proposed_losses[gen], pool.losses[gen]
            )
    gen = source == 1
    if gen.any():
        for j in range(pool.params.shape[1]):
            if pretrain_losses is not None:
                metrics[f"pool/generated_param{j}_vs_pretrain_loss_corr"] = safe_corr(
                    pool.params[gen, j], pretrain_losses[gen]
                )
            if pool.losses is not None:
                metrics[f"pool/generated_param{j}_vs_posttrain_loss_corr"] = safe_corr(
                    pool.params[gen, j], pool.losses[gen]
                )
    return metrics


def colormap_rgb(values: np.ndarray, cmap: str) -> np.ndarray:
    anchors = {
        "magma": np.asarray(
            [
                [0.001, 0.000, 0.014],
                [0.232, 0.060, 0.438],
                [0.550, 0.161, 0.506],
                [0.868, 0.287, 0.409],
                [0.995, 0.746, 0.518],
                [0.988, 0.998, 0.645],
            ],
            dtype=np.float32,
        ),
        "viridis": np.asarray(
            [
                [0.267, 0.005, 0.329],
                [0.283, 0.141, 0.458],
                [0.254, 0.265, 0.530],
                [0.207, 0.372, 0.553],
                [0.164, 0.471, 0.558],
                [0.128, 0.567, 0.551],
                [0.135, 0.659, 0.518],
                [0.267, 0.749, 0.441],
                [0.478, 0.821, 0.318],
                [0.741, 0.873, 0.150],
                [0.993, 0.906, 0.144],
            ],
            dtype=np.float32,
        ),
        "coolwarm": np.asarray(
            [
                [0.230, 0.299, 0.754],
                [0.554, 0.690, 0.996],
                [0.867, 0.864, 0.863],
                [0.957, 0.598, 0.477],
                [0.706, 0.016, 0.150],
            ],
            dtype=np.float32,
        ),
        "icefire": np.asarray(
            [
                [0.094, 0.110, 0.262],
                [0.050, 0.379, 0.596],
                [0.735, 0.925, 0.929],
                [0.973, 0.730, 0.489],
                [0.690, 0.083, 0.165],
                [0.122, 0.043, 0.084],
            ],
            dtype=np.float32,
        ),
    }[cmap]
    scaled = np.asarray(values, dtype=np.float32).clip(0.0, 1.0)
    pos = scaled * (len(anchors) - 1)
    left = np.floor(pos).astype(np.int64)
    right = np.clip(left + 1, 0, len(anchors) - 1)
    weight = (pos - left)[..., None]
    rgb = anchors[left] * (1.0 - weight) + anchors[right] * weight
    return (255.0 * rgb).clip(0, 255).astype(np.uint8)


def heatmap_image(values: np.ndarray, cmap: str = "coolwarm", center_zero: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if center_zero:
        scale = float(np.quantile(np.abs(values), 0.98))
        scale = max(scale, 1e-8)
        scaled = 0.5 + 0.5 * (values / scale)
    else:
        lo, hi = np.quantile(values, [0.02, 0.98])
        scaled = (values - lo) / (hi - lo + 1e-8)
    return colormap_rgb(scaled, cmap)


def stacked_example_image(examples: dict[str, np.ndarray]) -> np.ndarray:
    rows = []
    for i in range(examples["input"].shape[0]):
        rows.extend(
            [
                examples["input"][i],
                examples["target"][i],
                examples["prediction"][i],
                examples["error"][i],
            ]
        )
    return heatmap_image(np.stack(rows, axis=0), cmap="coolwarm", center_zero=True)


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
        payload["pool/loss_hist"] = wandb.Histogram(pool.losses)
    if pool.source is not None:
        source = pool.source.reshape(-1)
        for value, name in ((0, "uniform"), (1, "generated")):
            mask = source == value
            if not mask.any():
                continue
            if pool.losses is not None:
                payload[f"pool/{name}_posttrain_loss_hist"] = wandb.Histogram(pool.losses[mask])
            if pool.pretrain_losses is not None:
                payload[f"pool/{name}_pretrain_loss_hist"] = wandb.Histogram(pool.pretrain_losses[mask])
            for j in range(pool.params.shape[1]):
                payload[f"pool/{name}_param{j}_hist"] = wandb.Histogram(pool.params[mask, j])
            n_examples = min(24, mask.sum())
            idx = np.linspace(0, mask.sum() - 1, n_examples, dtype=np.int64)
            payload[f"pool/{name}_state_examples"] = wandb.Image(
                heatmap_image(pool.states[mask][idx, 0], cmap="coolwarm", center_zero=True),
                caption=f"{name} state samples, signed values",
            )
            payload[f"pool/{name}_state_abs_examples"] = wandb.Image(
                heatmap_image(np.abs(pool.states[mask][idx, 0]), cmap="magma", center_zero=False),
                caption=f"{name} state samples, absolute amplitude",
            )
    if pool.source is not None and pool.params.shape[1] > 1:
        rows = []
        source = pool.source.reshape(-1)
        for value, name in ((0, "uniform"), (1, "generated")):
            idx = np.where(source == value)[0]
            if len(idx) == 0:
                continue
            idx = idx[np.linspace(0, len(idx) - 1, min(128, len(idx)), dtype=np.int64)]
            for i in idx:
                row = [name, int(i), *pool.params[i].astype(float).tolist()]
                if pool.pretrain_losses is not None:
                    row.append(float(pool.pretrain_losses[i]))
                if pool.losses is not None:
                    row.append(float(pool.losses[i]))
                rows.append(row)
        columns = ["source", "index"] + [f"param{j}" for j in range(pool.params.shape[1])]
        if pool.pretrain_losses is not None:
            columns.append("pretrain_loss")
        if pool.losses is not None:
            columns.append("posttrain_loss")
        if rows:
            payload["pool/parameter_samples"] = wandb.Table(columns=columns, data=rows)
    n_pool_examples = min(32, len(pool))
    if n_pool_examples > 0:
        idx = np.linspace(0, len(pool) - 1, n_pool_examples, dtype=np.int64)
        payload["pool/state_examples"] = wandb.Image(
            heatmap_image(pool.states[idx, 0], cmap="coolwarm", center_zero=True),
            caption="pool state samples, signed values",
        )

    examples = one_step_examples(model, validation, device=device, n_examples=8)
    payload["val/one_step_examples"] = wandb.Image(
        stacked_example_image(examples),
        caption="input / target / prediction / signed error, repeated per validation case",
    )

    steps = min(cfg.validation.rollout_steps, validation.trajectory_steps)
    spacetime = rollout_spacetime_examples(model, validation, steps=steps, device=device, n_examples=2)
    for i in range(spacetime["truth"].shape[0]):
        payload[f"rollout_spacetime/truth_{i}"] = wandb.Image(
            heatmap_image(spacetime["truth"][i], cmap="coolwarm", center_zero=True),
            caption="truth spacetime, y=grid x=timestep",
        )
        payload[f"rollout_spacetime/prediction_{i}"] = wandb.Image(
            heatmap_image(spacetime["prediction"][i], cmap="coolwarm", center_zero=True),
            caption="prediction spacetime, y=grid x=timestep",
        )
        payload[f"rollout_spacetime/error_{i}"] = wandb.Image(
            heatmap_image(spacetime["error"][i], cmap="icefire", center_zero=True),
            caption="signed rollout error, y=grid x=timestep",
        )
        payload[f"rollout_spacetime/abs_error_{i}"] = wandb.Image(
            heatmap_image(np.abs(spacetime["error"][i]), cmap="magma", center_zero=False),
            caption="absolute rollout error, y=grid x=timestep",
        )

    if cfg.ddpm.enabled:
        loss_grid = (
            np.linspace(0.0, 1.0, 32, dtype=np.float32) if cfg.ddpm.mode == "conditional_loss" else None
        )
        gen_states, gen_params = generate_states_and_params(
            ddpm,
            32,
            cfg.ddpm.mode,
            loss_grid,
            pde,
            state_mean,
            state_std,
            np.random.default_rng(round_id + 1),
            device,
            batch_size=32,
        )
        payload["ddpm/sample_states"] = wandb.Image(
            heatmap_image(gen_states[:16, 0], cmap="coolwarm", center_zero=True),
            caption="DDPM generated state samples (physical units)",
        )
        payload["ddpm/sample_abs_states"] = wandb.Image(
            heatmap_image(np.abs(gen_states[:16, 0]), cmap="magma", center_zero=False),
            caption="DDPM generated state absolute amplitude",
        )
        payload.update(state_quality_metrics(gen_states, "ddpm/sample_states"))
        if cfg.ddpm.param_mode in ("condition", "generate"):
            uniform_ref = pde.sample_params_uniform(512, np.random.default_rng(round_id + 7))
            for j in range(gen_params.shape[1]):
                payload[f"ddpm/sample_param{j}_hist"] = wandb.Histogram(gen_params[:, j])
                payload[f"ddpm/uniform_param{j}_hist"] = wandb.Histogram(uniform_ref[:, j])

    series = rollout_error_series(model, validation, steps=steps, device=device, n_trajectories=256)
    hard_spacetime = rollout_spacetime_examples(
        model, validation, steps=steps, device=device, n_examples=2, hard=True
    )
    for i in range(hard_spacetime["truth"].shape[0]):
        payload[f"rollout_spacetime_hard/truth_{i}"] = wandb.Image(
            heatmap_image(hard_spacetime["truth"][i], cmap="coolwarm", center_zero=True),
            caption="hard validation truth spacetime, y=grid x=timestep",
        )
        payload[f"rollout_spacetime_hard/prediction_{i}"] = wandb.Image(
            heatmap_image(hard_spacetime["prediction"][i], cmap="coolwarm", center_zero=True),
            caption="hard validation prediction spacetime, y=grid x=timestep",
        )
        payload[f"rollout_spacetime_hard/error_{i}"] = wandb.Image(
            heatmap_image(hard_spacetime["error"][i], cmap="icefire", center_zero=True),
            caption="hard validation signed rollout error",
        )
        payload[f"rollout_spacetime_hard/abs_error_{i}"] = wandb.Image(
            heatmap_image(np.abs(hard_spacetime["error"][i]), cmap="magma", center_zero=False),
            caption="hard validation absolute rollout error",
        )
    payload["rollout/rmse_curve"] = wandb.plot.line_series(
        xs=series["step"].tolist(),
        ys=[series["rmse_mean"].tolist(), series["rmse_p95"].tolist(), series["rmse_p99"].tolist()],
        keys=["mean", "p95", "p99"],
        title="Rollout RMSE",
        xname="step",
    )
    payload["rollout/nrmse_curve"] = wandb.plot.line_series(
        xs=series["step"].tolist(),
        ys=[
            series["nrmse_mean"].tolist(),
            series["nrmse_p95"].tolist(),
            series["nrmse_p99"].tolist(),
        ],
        keys=["mean", "p95", "p99"],
        title="Rollout NRMSE",
        xname="step",
    )
    run.log(payload, step=step)


def make_epoch_callback(run, model, validation, cfg, device, round_id: int, prefix: str):
    if run is None:
        return None

    def _callback(metrics: dict[str, float], epoch: int) -> None:
        epoch_metrics = {f"{prefix}/{k.split('/', 1)[-1]}": v for k, v in metrics.items()}
        epoch_metrics["round"] = round_id
        epoch_metrics["epoch"] = epoch
        if prefix == "surrogate_epoch":
            val = evaluate_one_step_validation(
                model,
                validation,
                cfg.validation.quantiles,
                device,
                n_trajectories=cfg.validation.epoch_n_trajectories,
                prefix="surrogate_epoch_val",
            )
            epoch_metrics.update(val)
        stride = (
            max(1, cfg.surrogate.epochs_per_round)
            + (max(1, cfg.ddpm.train_epochs) if cfg.ddpm.enabled else 0)
            + 10
        )
        offset = 0 if prefix == "surrogate_epoch" else max(1, cfg.surrogate.epochs_per_round)
        run.log(epoch_metrics, step=round_id * stride + offset + epoch)

    return _callback


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir/checkpoint_latest.pt")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing checkpoint")
    parser.add_argument(
        "--create-validation-only",
        action="store_true",
        help="Create or validate the configured validation dataset and exit",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set seed=101 --set pool.rounds=4",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for expression in args.overrides:
        apply_override(cfg, expression)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    pde = PDE1D(cfg.pde)
    validation = load_or_create_validation(cfg, pde, out)
    if args.create_validation_only:
        return

    # Validation-set statistics used to z-normalize states for DDPM training/generation,
    # so generated states stay within the physical range of the data.
    state_mean = float(validation.trajectories.mean())
    state_std = float(validation.trajectories.std()) + 1e-8
    print(f"state z-norm from validation: mean={state_mean:.4f} std={state_std:.4f}")

    device = choose_device(cfg.device)
    model = build_surrogate(
        resolution=cfg.pde.resolution,
        hidden=cfg.surrogate.hidden,
        depth=cfg.surrogate.depth,
        ensemble_size=cfg.surrogate.ensemble_size,
        param_dim=len(pde.param_ranges),
        model_name=cfg.surrogate.model,
        difference_weight=cfg.surrogate.difference_weight,
    ).to(device)
    ddpm = DDPM1D(
        resolution=cfg.pde.resolution,
        hidden=cfg.ddpm.hidden,
        steps=cfg.ddpm.steps,
        loss_conditional=cfg.ddpm.mode == "conditional_loss",
        param_dim=len(pde.param_ranges),
        param_mode=cfg.ddpm.param_mode,
    ).to(device)
    checkpoint = out / "checkpoint_latest.pt"
    start_round = 0
    pool = make_uniform_pool(pde, cfg.pool.n_trajectories, cfg.pool.trajectory_steps, rng)
    history: list[dict[str, float]] = []
    wandb_run_id = uuid.uuid4().hex
    if not cfg.ddpm.enabled and cfg.pool.uniform_fraction < 1.0:
        raise ValueError("ddpm.enabled=false requires pool.uniform_fraction=1.0")
    if args.resume and not args.fresh and checkpoint.exists():
        start_round, pool, history, wandb_run_id = load_checkpoint(checkpoint, model, ddpm, rng, device)
        print(f"resumed from {checkpoint} at round {start_round}")
    run = maybe_wandb(cfg, wandb_run_id)

    for round_id in range(start_round, cfg.pool.rounds):
        if round_id > 0:
            pool = make_mixed_pool(
                pde=pde,
                ddpm=ddpm,
                previous=pool,
                n_traj=cfg.pool.n_trajectories,
                steps=cfg.pool.trajectory_steps,
                uniform_fraction=cfg.pool.uniform_fraction,
                ddpm_mode=cfg.ddpm.mode,
                rng=rng,
                device=device,
                state_mean=state_mean,
                state_std=state_std,
            )
        pretrain_losses = compute_transition_losses(model, pool, cfg.surrogate.batch_size, device)
        pool.pretrain_losses = pretrain_losses
        debug_metrics = pool_debug_metrics(pool, pretrain_losses)
        train_metrics = train_surrogate(
            model=model,
            pool=pool,
            epochs=cfg.surrogate.epochs_per_round,
            batch_size=cfg.surrogate.batch_size,
            lr=cfg.surrogate.lr,
            weight_decay=cfg.surrogate.weight_decay,
            device=device,
            epoch_callback=make_epoch_callback(
                run, model, validation, cfg, device, round_id, "surrogate_epoch"
            ),
            round_id=round_id,
        )
        ddpm_metrics = {}
        if cfg.ddpm.enabled:
            ddpm_metrics = train_ddpm(
                ddpm=ddpm,
                pool=pool,
                epochs=cfg.ddpm.train_epochs,
                batch_size=cfg.ddpm.batch_size,
                lr=cfg.ddpm.lr,
                device=device,
                state_mean=state_mean,
                state_std=state_std,
                pde=pde,
                mode=cfg.ddpm.mode,
                param_loss_weight=cfg.ddpm.param_loss_weight,
                epoch_callback=make_epoch_callback(
                    run, model, validation, cfg, device, round_id, "ddpm_epoch"
                ),
                round_id=round_id,
            )
        eval_metrics = evaluate_validation(
            model=model,
            validation=validation,
            rollout_steps=cfg.validation.rollout_steps,
            quantiles=cfg.validation.quantiles,
            device=device,
        )
        metrics = {
            "round": round_id,
            "pool/n_samples": len(pool),
            "pool/loss_mean": float(np.mean(pool.losses)) if pool.losses is not None else 0.0,
            "pool/loss_p90": float(np.quantile(pool.losses, 0.9)) if pool.losses is not None else 0.0,
            **debug_metrics,
            **train_metrics,
            **ddpm_metrics,
            **eval_metrics,
        }
        metrics.update(pool_debug_metrics(pool, pretrain_losses))
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))
        stride = (
            max(1, cfg.surrogate.epochs_per_round)
            + (max(1, cfg.ddpm.train_epochs) if cfg.ddpm.enabled else 0)
            + 10
        )
        round_step = round_id * stride + stride - 2
        if run is not None:
            run.log(metrics, step=round_step)
            try:
                log_wandb_diagnostics(
                    run, model, ddpm, pool, validation, cfg, device, round_id, round_step + 1,
                    pde, state_mean, state_std,
                )
            except Exception as exc:
                print(f"wandb diagnostics failed at round {round_id}: {exc}", file=sys.stderr)
        np.savez_compressed(
            out / f"pool_round_{round_id}.npz",
            states=pool.states,
            params=pool.params,
            next_states=pool.next_states,
            losses=pool.losses,
            source=pool.source,
            proposed_losses=pool.proposed_losses,
            pretrain_losses=pool.pretrain_losses,
        )
        torch.save(model.state_dict(), out / "surrogate.pt")
        torch.save(ddpm.state_dict(), out / "ddpm.pt")
        save_checkpoint(checkpoint, round_id, model, ddpm, pool, history, rng, wandb_run_id)

    (out / "history.json").write_text(json.dumps(history, indent=2))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
