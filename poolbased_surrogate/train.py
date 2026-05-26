from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import StateLossDataset, TransitionDataset, TransitionPool
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
) -> dict[str, float]:
    model.train()
    opts = [
        torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        for m in model.models
    ]
    loader = DataLoader(TransitionDataset(pool), batch_size=batch_size, shuffle=True, drop_last=False)
    last_losses = []
    for _ in range(epochs):
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
    mode: str = "conditional_loss",
) -> dict[str, float]:
    if pool.losses is None:
        raise ValueError("Pool losses required before DDPM training.")
    ddpm.train()
    losses = normalize_losses(pool.losses)
    loader = DataLoader(StateLossDataset(pool.states, losses), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(ddpm.parameters(), lr=lr)
    last = []
    for _ in range(epochs):
        for state, loss_value in loader:
            opt.zero_grad(set_to_none=True)
            if mode == "conditional_loss":
                loss = ddpm.training_loss(state.to(device), loss_value.to(device))
            elif mode == "weighted_unconditional":
                weights = 0.05 + loss_value.to(device).view(-1)
                loss = ddpm.training_loss(state.to(device), None, sample_weight=weights)
            else:
                raise ValueError(f"Unknown ddpm mode {mode}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            last.append(float(loss.detach().cpu()))
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
