from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import TransitionPool
from .models.surrogate import EnsembleSurrogate
from .pde import PDE1D


def quantile_label(q: float) -> str:
    return f"{100.0 * q:g}"


@dataclass
class ValidationData:
    states0: np.ndarray
    params: np.ndarray
    trajectories: np.ndarray

    @property
    def n_trajectories(self) -> int:
        return int(self.states0.shape[0])

    @property
    def trajectory_steps(self) -> int:
        return int(self.trajectories.shape[1] - 1)


def create_validation_data(
    pde: PDE1D,
    n_trajectories: int,
    trajectory_steps: int,
    seed: int,
) -> ValidationData:
    params = pde.sample_params_halton(n_trajectories, seed=seed)
    states0 = pde.sample_ic_halton(n_trajectories, seed=seed + 1000)
    trajectories = pde.simulate(states0, params, trajectory_steps)
    return ValidationData(states0=states0, params=params, trajectories=trajectories)


def save_validation_data(path: Path, data: ValidationData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states0=data.states0,
        params=data.params,
        trajectories=data.trajectories,
    )


def load_validation_data(path: Path) -> ValidationData:
    with np.load(path) as payload:
        return ValidationData(
            states0=payload["states0"],
            params=payload["params"],
            trajectories=payload["trajectories"],
        )


@torch.no_grad()
def evaluate(
    model: EnsembleSurrogate,
    pde: PDE1D,
    n_trajectories: int,
    trajectory_steps: int,
    rollout_steps: int,
    quantiles: list[float],
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    validation = create_validation_data(pde, n_trajectories, trajectory_steps, seed)
    return evaluate_validation(model, validation, rollout_steps, quantiles, device)


@torch.no_grad()
def evaluate_validation(
    model: EnsembleSurrogate,
    validation: ValidationData,
    rollout_steps: int,
    quantiles: list[float],
    device: torch.device,
) -> dict[str, float]:
    pool = TransitionPool.from_trajectories(validation.trajectories, validation.params)
    pred = batched_predict(model, pool.states, pool.params, device)
    err = pred - pool.next_states
    mse = np.mean(err**2, axis=(1, 2))
    rmse = np.sqrt(mse)
    denom = np.sqrt(np.mean(pool.next_states**2, axis=(1, 2))) + 1e-8
    nrmse = rmse / denom
    metrics = {
        "val/rmse_mean": float(rmse.mean()),
        "val/nrmse_mean": float(nrmse.mean()),
    }
    for q in quantiles:
        label = quantile_label(q)
        metrics[f"val/rmse_p{label}"] = float(np.quantile(rmse, q))
        metrics[f"val/nrmse_p{label}"] = float(np.quantile(nrmse, q))
    difficulty = np.mean(pool.next_states**2, axis=(1, 2))
    for q in quantiles:
        cutoff = np.quantile(difficulty, q)
        mask = difficulty <= cutoff
        if mask.any():
            metrics[f"val/rmse_q{q:g}"] = float(rmse[mask].mean())
            metrics[f"val/nrmse_q{q:g}"] = float(nrmse[mask].mean())
    metrics.update(
        rollout_metrics_from_truth(
            model,
            validation.states0,
            validation.params,
            validation.trajectories,
            min(rollout_steps, validation.trajectory_steps),
            device,
        )
    )
    return metrics


@torch.no_grad()
def evaluate_one_step_validation(
    model: EnsembleSurrogate,
    validation: ValidationData,
    quantiles: list[float],
    device: torch.device,
    n_trajectories: int | None = None,
    prefix: str = "epoch_val",
) -> dict[str, float]:
    if n_trajectories is not None and n_trajectories < validation.n_trajectories:
        idx = np.linspace(0, validation.n_trajectories - 1, n_trajectories, dtype=np.int64)
        trajectories = validation.trajectories[idx]
        params = validation.params[idx]
    else:
        trajectories = validation.trajectories
        params = validation.params
    pool = TransitionPool.from_trajectories(trajectories, params)
    pred = batched_predict(model, pool.states, pool.params, device)
    err = pred - pool.next_states
    rmse = np.sqrt(np.mean(err**2, axis=(1, 2)))
    denom = np.sqrt(np.mean(pool.next_states**2, axis=(1, 2))) + 1e-8
    nrmse = rmse / denom
    metrics = {
        f"{prefix}/rmse_mean": float(rmse.mean()),
        f"{prefix}/nrmse_mean": float(nrmse.mean()),
    }
    for q in quantiles:
        label = quantile_label(q)
        metrics[f"{prefix}/rmse_p{label}"] = float(np.quantile(rmse, q))
        metrics[f"{prefix}/nrmse_p{label}"] = float(np.quantile(nrmse, q))
    return metrics


@torch.no_grad()
def batched_predict(
    model: EnsembleSurrogate,
    states: np.ndarray,
    params: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(states), batch_size):
        state = torch.from_numpy(states[i : i + batch_size]).to(device)
        par = torch.from_numpy(params[i : i + batch_size]).to(device)
        out.append(model(state, par).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def one_step_examples(
    model: EnsembleSurrogate,
    validation: ValidationData,
    device: torch.device,
    n_examples: int = 8,
) -> dict[str, np.ndarray]:
    n = min(n_examples, validation.n_trajectories)
    idx = np.linspace(0, validation.n_trajectories - 1, n, dtype=np.int64)
    states = validation.trajectories[idx, 0]
    target = validation.trajectories[idx, 1]
    params = validation.params[idx]
    pred = batched_predict(model, states, params, device)
    return {
        "input": states[:, 0],
        "target": target[:, 0],
        "prediction": pred[:, 0],
        "error": (pred - target)[:, 0],
    }


@torch.no_grad()
def rollout_error_series(
    model: EnsembleSurrogate,
    validation: ValidationData,
    steps: int,
    device: torch.device,
    n_trajectories: int = 256,
) -> dict[str, np.ndarray]:
    n = min(n_trajectories, validation.n_trajectories)
    idx = np.linspace(0, validation.n_trajectories - 1, n, dtype=np.int64)
    states0 = validation.states0[idx]
    params = validation.params[idx]
    target = validation.trajectories[idx, 1 : steps + 1]
    state = torch.from_numpy(states0).to(device)
    par = torch.from_numpy(params).to(device)
    preds = []
    model.eval()
    for _ in range(steps):
        state = model(state, par)
        preds.append(state.cpu().numpy())
    rollout = np.stack(preds, axis=1)
    rmse = np.sqrt(np.mean((rollout - target) ** 2, axis=(2, 3)))
    nrmse = rmse / (np.sqrt(np.mean(target**2, axis=(2, 3))) + 1e-8)
    return {
        "step": np.arange(1, steps + 1),
        "rmse_mean": rmse.mean(axis=0),
        "rmse_p95": np.quantile(rmse, 0.95, axis=0),
        "rmse_p99": np.quantile(rmse, 0.99, axis=0),
        "nrmse_mean": nrmse.mean(axis=0),
        "nrmse_p95": np.quantile(nrmse, 0.95, axis=0),
        "nrmse_p99": np.quantile(nrmse, 0.99, axis=0),
    }


@torch.no_grad()
def rollout_spacetime_examples(
    model: EnsembleSurrogate,
    validation: ValidationData,
    steps: int,
    device: torch.device,
    n_examples: int = 2,
    hard: bool = False,
) -> dict[str, np.ndarray]:
    n = min(n_examples, validation.n_trajectories)
    if hard:
        # Model-independent difficulty so the SAME example set is shown every round and
        # every run (comparable figures): rank by mean spatial total-variation of the
        # ground-truth trajectory over the rollout horizon. Higher TV = rougher = harder.
        gt = validation.trajectories[:, : steps + 1, 0]  # (N, T+1, L)
        gt_tv = np.abs(np.diff(gt, axis=-1)).sum(axis=-1).mean(axis=1)  # (N,)
        idx = np.sort(np.argsort(gt_tv)[-n:])
    else:
        idx = np.linspace(0, validation.n_trajectories - 1, n, dtype=np.int64)
    states0 = validation.states0[idx]
    params = validation.params[idx]
    truth = validation.trajectories[idx, : steps + 1, 0]
    state = torch.from_numpy(states0).to(device)
    par = torch.from_numpy(params).to(device)
    preds = [states0]
    model.eval()
    for _ in range(steps):
        state = model(state, par)
        preds.append(state.cpu().numpy())
    pred = np.concatenate(preds, axis=1)
    return {
        "truth": np.transpose(truth, (0, 2, 1)),
        "prediction": np.transpose(pred, (0, 2, 1)),
        "error": np.transpose(pred - truth, (0, 2, 1)),
    }


@torch.no_grad()
def rollout_metrics(
    model: EnsembleSurrogate,
    pde: PDE1D,
    states0: np.ndarray,
    params: np.ndarray,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    truth = pde.simulate(states0, params, steps)
    return rollout_metrics_from_truth(model, states0, params, truth, steps, device)


@torch.no_grad()
def rollout_metrics_from_truth(
    model: EnsembleSurrogate,
    states0: np.ndarray,
    params: np.ndarray,
    truth: np.ndarray,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    state = torch.from_numpy(states0).to(device)
    par = torch.from_numpy(params).to(device)
    preds = []
    model.eval()
    for _ in range(steps):
        state = model(state, par)
        preds.append(state.cpu().numpy())
    rollout = np.stack(preds, axis=1)
    target = truth[:, 1 : steps + 1]
    per_traj_rmse_t = np.sqrt(np.mean((rollout - target) ** 2, axis=(2, 3)))
    per_traj_nrmse_t = per_traj_rmse_t / (np.sqrt(np.mean(target**2, axis=(2, 3))) + 1e-8)
    rmse_t = np.sqrt(np.mean((rollout - target) ** 2, axis=(0, 2, 3)))
    nrmse_t = rmse_t / (np.sqrt(np.mean(target**2, axis=(0, 2, 3))) + 1e-8)
    metrics = {
        "rollout/rmse_final": float(rmse_t[-1]),
        "rollout/rmse_mean": float(rmse_t.mean()),
        "rollout/nrmse_final": float(nrmse_t[-1]),
        "rollout/nrmse_mean": float(nrmse_t.mean()),
    }
    for q in (0.95, 0.99):
        label = quantile_label(q)
        metrics[f"rollout/rmse_final_p{label}"] = float(np.quantile(per_traj_rmse_t[:, -1], q))
        metrics[f"rollout/rmse_mean_p{label}"] = float(np.quantile(per_traj_rmse_t.mean(axis=1), q))
        metrics[f"rollout/nrmse_final_p{label}"] = float(np.quantile(per_traj_nrmse_t[:, -1], q))
        metrics[f"rollout/nrmse_mean_p{label}"] = float(np.quantile(per_traj_nrmse_t.mean(axis=1), q))
    return metrics
