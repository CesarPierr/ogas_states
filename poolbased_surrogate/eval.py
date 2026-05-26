from __future__ import annotations

import numpy as np
import torch

from .data import TransitionPool
from .models.surrogate import EnsembleSurrogate
from .pde import PDE1D


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
    params = pde.sample_params_halton(n_trajectories, seed=seed)
    states0 = pde.sample_ic_halton(n_trajectories, seed=seed + 1000)
    traj = pde.simulate(states0, params, trajectory_steps)
    pool = TransitionPool.from_trajectories(traj, params)
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
    difficulty = np.mean(pool.next_states**2, axis=(1, 2))
    for q in quantiles:
        cutoff = np.quantile(difficulty, q)
        mask = difficulty <= cutoff
        if mask.any():
            metrics[f"val/rmse_q{q:g}"] = float(rmse[mask].mean())
            metrics[f"val/nrmse_q{q:g}"] = float(nrmse[mask].mean())
    metrics.update(rollout_metrics(model, pde, states0, params, min(rollout_steps, trajectory_steps), device))
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
def rollout_metrics(
    model: EnsembleSurrogate,
    pde: PDE1D,
    states0: np.ndarray,
    params: np.ndarray,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    truth = pde.simulate(states0, params, steps)
    state = torch.from_numpy(states0).to(device)
    par = torch.from_numpy(params).to(device)
    preds = []
    model.eval()
    for _ in range(steps):
        state = model(state, par)
        preds.append(state.cpu().numpy())
    rollout = np.stack(preds, axis=1)
    target = truth[:, 1:]
    rmse_t = np.sqrt(np.mean((rollout - target) ** 2, axis=(0, 2, 3)))
    nrmse_t = rmse_t / (np.sqrt(np.mean(target**2, axis=(0, 2, 3))) + 1e-8)
    return {
        "rollout/rmse_final": float(rmse_t[-1]),
        "rollout/rmse_mean": float(rmse_t.mean()),
        "rollout/nrmse_final": float(nrmse_t[-1]),
        "rollout/nrmse_mean": float(nrmse_t.mean()),
    }
