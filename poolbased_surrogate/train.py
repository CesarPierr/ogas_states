from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import TransitionDataset, TransitionPool
from .models.ddpm import DDPM1D
from .models.surrogate import EnsembleSurrogate


def assign_quantile_bins(losses: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin (already-normalised) losses into n_bins equal-frequency quantile bins.

    Returns (labels in {0..n_bins-1}, edges of length n_bins+1).
    """
    losses = np.asarray(losses, dtype=np.float32).reshape(-1)
    edges = np.quantile(losses, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = edges[-1] * (1.0 + 1e-6)
    labels = np.digitize(losses, edges[1:], right=False).clip(0, n_bins - 1).astype(np.int64)
    return labels, edges.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample_quantile_labels(
    strategy: str, n: int, n_quantiles: int, rng: np.random.Generator, temp: float = 0.5
) -> np.ndarray:
    """Draw n target quantile-bin labels per a difficulty-sampling law.

    Strategies (q = n_quantiles): top1/2/3/5 = uniform over the hardest k bins,
    top_half = uniform over the upper half, exp_bias = softmax weighting toward
    hard bins (temperature `temp`), uniform = all bins equally.
    """
    q = int(n_quantiles)
    s = str(strategy).lower()
    if s == "top1":
        return np.full(n, q - 1, dtype=np.int64)
    if s in ("top2", "top3", "top5"):
        k = int(s[3:])
        return rng.choice(np.arange(max(0, q - k), q), size=n).astype(np.int64)
    if s == "top_half":
        return rng.integers(q // 2, q, size=n).astype(np.int64)
    if s == "exp_bias":
        return rng.choice(q, size=n, p=_softmax(np.arange(q, dtype=float) * float(temp))).astype(np.int64)
    if s == "uniform":
        return rng.integers(0, q, size=n).astype(np.int64)
    raise ValueError(f"Unknown quantile sample strategy {strategy!r}")


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
    seed: int | None = None,
) -> dict[str, float]:
    if seed is not None:
        torch.manual_seed(seed)
    model.train()

    opts = [
        torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        for m in model.models
    ]
    loader_generator = None
    if seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    loader = DataLoader(
        TransitionDataset(pool),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
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
    if getattr(model, "n_models", 1) > 1:
        pool.uncertainty = compute_transition_uncertainty(model, pool, batch_size, device)
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


@torch.no_grad()
def compute_transition_uncertainty(
    model: EnsembleSurrogate,
    pool: TransitionPool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Per-transition epistemic uncertainty = mean variance across ensemble members.

    This is the *reducible* part of the error (query-by-committee / approx-BALD signal),
    as opposed to compute_transition_losses which also includes irreducible/aleatoric
    error from model misspecification on a chaotic system (see docs/theory.md §3.3).
    """
    model.eval()
    out = []
    loader = DataLoader(TransitionDataset(pool), batch_size=batch_size, shuffle=False, drop_last=False)
    for state, params, _ in loader:
        out.append(model.uncertainty(state.to(device), params.to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


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
    loss_metric: str = "mse",
    difficulty_signal: str = "loss",
    param_loss_weight: float = 1.0,
    loss_condition_scale: float = 1.0,
    sample_weight_floor: float = 0.05,
    sample_weight_power: float = 1.0,
    epoch_callback=None,
    round_id: int = 0,
) -> dict[str, float]:
    if pool.losses is None:
        raise ValueError("Pool losses required before DDPM training.")
    if mode not in ("conditional_loss", "weighted_unconditional", "conditional_quantile"):
        raise ValueError(f"Unknown ddpm mode {mode}")
    ddpm.train()
    losses = normalize_losses(pool_difficulty_signal(pool, loss_metric, difficulty_signal))
    state_norm = ((pool.states - state_mean) / state_std).astype(np.float32)
    needs_params = ddpm.use_param_cond or ddpm.generate_params
    if needs_params:
        param_scaled = (2.0 * pde.normalize_params(pool.params) - 1.0).astype(np.float32)
    else:
        param_scaled = np.zeros((len(pool), max(1, ddpm.param_dim)), dtype=np.float32)

    # Quantile-bin conditioning: bin this round's pool losses, store edges on the
    # model so generation maps realized losses back to the same bins, and use a
    # balanced sampler so every difficulty bin contributes equal signal.
    quantile_mode = mode == "conditional_quantile"
    if quantile_mode:
        if ddpm.n_quantiles <= 0:
            raise ValueError("mode=conditional_quantile requires ddpm.n_quantiles > 0")
        qbins, qedges = assign_quantile_bins(losses, ddpm.n_quantiles)
        ddpm.quantile_edges = qedges  # consumed by the generation step
        qbins_t = torch.from_numpy(qbins)
    else:
        qbins_t = torch.zeros(len(state_norm), dtype=torch.int64)

    dataset = TensorDataset(
        torch.from_numpy(state_norm),
        torch.from_numpy(losses.reshape(-1, 1).astype(np.float32)),
        torch.from_numpy(param_scaled),
        qbins_t,
    )
    if quantile_mode:
        counts = np.bincount(qbins, minlength=ddpm.n_quantiles).astype(np.float64)
        weights = (1.0 / np.clip(counts, 1.0, None))[qbins]
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(ddpm.parameters(), lr=lr)
    last = []
    loss_condition_scale = float(loss_condition_scale)
    sample_weight_floor = float(sample_weight_floor)
    sample_weight_power = float(sample_weight_power)
    for epoch in range(epochs):
        epoch_losses = []
        for state, loss_value, params, qbin in loader:
            opt.zero_grad(set_to_none=True)
            norm_loss = loss_value.to(device)
            loss_cond = norm_loss * loss_condition_scale if mode == "conditional_loss" else None
            sample_weight = (
                sample_weight_floor + torch.clamp(norm_loss.view(-1), min=0.0) ** sample_weight_power
                if mode == "weighted_unconditional"
                else None
            )
            quantile_label = qbin.to(device) if quantile_mode else None
            params_arg = params.to(device) if needs_params else None
            loss = ddpm.training_loss(
                state.to(device),
                loss=loss_cond,
                params=params_arg,
                sample_weight=sample_weight,
                param_loss_weight=param_loss_weight,
                quantile_label=quantile_label,
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


def transform_transition_losses(losses: np.ndarray, metric: str = "mse") -> np.ndarray:
    losses = np.asarray(losses, dtype=np.float32)
    metric = str(metric or "mse").lower()
    if metric == "mse":
        return losses
    if metric in ("rmse", "nrmse"):
        # "nrmse" requires the per-transition target amplitude to normalise; when that
        # context is unavailable (loss histograms, conditional_loss proposals) we fall
        # back to plain RMSE. The amplitude-normalised version lives in pool_difficulty().
        return np.sqrt(np.maximum(losses, 0.0)).astype(np.float32)
    raise ValueError(f"Unknown loss metric {metric!r}; expected 'mse', 'rmse' or 'nrmse'.")


def pool_difficulty(raw_losses: np.ndarray, next_states: np.ndarray, metric: str = "rmse") -> np.ndarray:
    """Per-transition difficulty used for quantile binning / realized-bin assignment.

    metric='nrmse' divides each transition's RMSE by its target RMS amplitude (relative
    error, so low-amplitude/hard states rank higher); 'mse'/'rmse' use the absolute error.
    """
    raw_losses = np.asarray(raw_losses, dtype=np.float32)
    if str(metric or "").lower() == "nrmse":
        rmse = np.sqrt(np.maximum(raw_losses, 0.0))
        denom = np.sqrt(np.mean(np.asarray(next_states, dtype=np.float32) ** 2, axis=(1, 2))) + 1e-8
        return (rmse / denom).astype(np.float32)
    return transform_transition_losses(raw_losses, metric)


def pool_difficulty_signal(pool: TransitionPool, loss_metric: str, difficulty_signal: str) -> np.ndarray:
    """Raw per-transition difficulty used to bin the pool for quantile conditioning.

    difficulty_signal='ensemble_var' uses the ensemble disagreement (reducible/epistemic
    uncertainty); 'loss' uses the surrogate one-step error under loss_metric. Falls back to
    the loss signal if uncertainty is unavailable (e.g. ensemble_size == 1).
    """
    if str(difficulty_signal or "loss").lower() == "ensemble_var":
        if pool.uncertainty is not None:
            return np.asarray(pool.uncertainty, dtype=np.float32)
        # No ensemble -> no disagreement signal; degrade gracefully to the loss signal.
    return pool_difficulty(pool.losses, pool.next_states, loss_metric)


def realized_difficulty(
    gen_losses: np.ndarray,
    gen_uncertainty: np.ndarray | None,
    next_states: np.ndarray,
    loss_metric: str,
    difficulty_signal: str,
) -> np.ndarray:
    """Difficulty of generated states for realized-bin assignment in generator_eval,
    using the same signal as training so realized bins align with the stored edges."""
    if str(difficulty_signal or "loss").lower() == "ensemble_var" and gen_uncertainty is not None:
        return np.asarray(gen_uncertainty, dtype=np.float32)
    return pool_difficulty(gen_losses, next_states, loss_metric)


def propose_losses(losses: np.ndarray, n: int, rng: np.random.Generator, metric: str = "mse") -> np.ndarray:
    losses = normalize_losses(transform_transition_losses(losses, metric))
    if len(losses) == 0:
        return rng.uniform(0.5, 1.0, size=(n, 1)).astype(np.float32)
    idx = rng.integers(0, len(losses), size=n)
    sampled = losses[idx]
    noise = rng.normal(0.0, 0.05, size=n).astype(np.float32)
    return np.maximum(sampled + noise, 0.0).reshape(n, 1).astype(np.float32)
