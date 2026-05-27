from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import TransitionDataset, TransitionPool
from .models.ddpm import DDPM1D
from .models.surrogate import EnsembleSurrogate


def train_surrogate(
    model: EnsembleSurrogate,
    pool: TransitionPool,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    epoch_callback=None,
    round_id: int = 0,
) -> dict[str, float]:
    model.train()
    opts = [
        torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        for m in model.models
    ]
    loader = DataLoader(TransitionDataset(pool), batch_size=batch_size, shuffle=True, drop_last=False)
    last_losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for state, params, target in loader:
            state = state.to(device)
            params = params.to(device)
            target = target.to(device)
            for model_i, opt in zip(model.models, opts):
                opt.zero_grad(set_to_none=True)
                pred = model_i(state, params)
                loss = torch.mean((pred - target) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_i.parameters(), 1.0)
                opt.step()
            last_losses.append(float(loss.detach().cpu()))
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch_callback is not None:
            epoch_callback(
                {
                    "surrogate/epoch_loss": float(np.mean(epoch_losses)),
                    "surrogate/epoch": epoch,
                    "surrogate/global_epoch": round_id * epochs + epoch,
                },
                epoch,
            )
            model.train()
    pool.losses = compute_transition_losses(model, pool, batch_size, device)
    return {"train/loss": float(np.mean(last_losses[-max(1, len(loader)) :]))}


@torch.no_grad()
def compute_transition_losses(
    model: EnsembleSurrogate,
    pool: TransitionPool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    losses = []
    loader = DataLoader(TransitionDataset(pool), batch_size=batch_size, shuffle=False, drop_last=False)
    for state, params, target in loader:
        pred = model(state.to(device), params.to(device))
        mse = torch.mean((pred - target.to(device)) ** 2, dim=(1, 2))
        losses.append(mse.cpu().numpy())
    return np.concatenate(losses).astype(np.float32)


def train_ddpm(
    ddpm: DDPM1D,
    pool: TransitionPool,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    state_mean: float,
    state_std: float,
    pde,
    mode: str = "conditional_loss",
    param_loss_weight: float = 1.0,
    epoch_callback=None,
    round_id: int = 0,
) -> dict[str, float]:
    if pool.losses is None:
        raise ValueError("Pool losses required before DDPM training.")
    ddpm.train()
    losses = normalize_losses(pool.losses)
    state_norm = ((pool.states - state_mean) / state_std).astype(np.float32)
    needs_params = ddpm.use_param_cond or ddpm.generate_params
    if needs_params:
        param_scaled = (2.0 * pde.normalize_params(pool.params) - 1.0).astype(np.float32)
    else:
        param_scaled = np.zeros((len(pool), max(1, ddpm.param_dim)), dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(state_norm),
        torch.from_numpy(losses.reshape(-1, 1).astype(np.float32)),
        torch.from_numpy(param_scaled),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(ddpm.parameters(), lr=lr)
    last = []
    for epoch in range(epochs):
        epoch_losses = []
        for state, loss_value, params in loader:
            opt.zero_grad(set_to_none=True)
            loss_cond = loss_value.to(device) if mode == "conditional_loss" else None
            sample_weight = 0.05 + loss_value.to(device).view(-1) if mode == "weighted_unconditional" else None
            params_arg = params.to(device) if needs_params else None
            if mode not in ("conditional_loss", "weighted_unconditional"):
                raise ValueError(f"Unknown ddpm mode {mode}")
            loss = ddpm.training_loss(
                state.to(device),
                loss=loss_cond,
                params=params_arg,
                sample_weight=sample_weight,
                param_loss_weight=param_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            last.append(float(loss.detach().cpu()))
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch_callback is not None:
            epoch_callback(
                {
                    "ddpm/epoch_loss": float(np.mean(epoch_losses)),
                    "ddpm/epoch": epoch,
                    "ddpm/global_epoch": round_id * epochs + epoch,
                },
                epoch,
            )
    return {"ddpm/loss": float(np.mean(last[-max(1, len(loader)) :]))}


def normalize_losses(losses: np.ndarray) -> np.ndarray:
    losses = np.asarray(losses, dtype=np.float32)
    lo = np.quantile(losses, 0.05)
    hi = np.quantile(losses, 0.95)
    return ((losses - lo) / (hi - lo + 1e-8)).clip(0.0, 1.5).astype(np.float32)


def propose_losses(losses: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    losses = normalize_losses(losses)
    if len(losses) == 0:
        return rng.uniform(0.5, 1.0, size=(n, 1)).astype(np.float32)
    idx = rng.integers(0, len(losses), size=n)
    sampled = losses[idx]
    noise = rng.normal(0.0, 0.05, size=n).astype(np.float32)
    return np.maximum(sampled + noise, 0.0).reshape(n, 1).astype(np.float32)
