"""Pool construction: uniform sampling and the difficulty-conditioned mixed pool.

Each round replaces the training pool with a fresh fixed-size pool that mixes
freshly simulated uniform trajectories with solver-labelled states drawn from the
generator. The split is controlled by ``uniform_fraction`` (see ``make_mixed_pool``).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from .data import TransitionPool
from .models.ddpm import DDPM1D
from .pde import PDE1D
from .train import (
    normalize_losses,
    propose_losses,
    sample_quantile_labels,
    transform_transition_losses,
)


def make_uniform_pool(pde: PDE1D, n_traj: int, steps: int, rng: np.random.Generator) -> TransitionPool:
    params = pde.sample_params_uniform(n_traj, rng)
    states0 = pde.sample_ic_uniform(n_traj, rng)
    traj = pde.simulate(states0, params, steps)
    return TransitionPool.from_trajectories(traj, params)


def finite_rows(*arrays: np.ndarray) -> np.ndarray:
    mask = None
    for array in arrays:
        arr = np.asarray(array)
        row_mask = np.isfinite(arr.reshape(arr.shape[0], -1)).all(axis=1)
        mask = row_mask if mask is None else mask & row_mask
    if mask is None:
        raise ValueError("finite_rows requires at least one array")
    return mask


def safe_step_transitions(
    pde: PDE1D,
    states: np.ndarray,
    params: np.ndarray,
    batch_size: int = 512,
    progress_label: str | None = None,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    params = np.asarray(params, dtype=np.float32)
    next_states = np.full_like(states, np.nan, dtype=np.float32)
    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        chunk_t0 = time.perf_counter()
        try:
            stepped = pde.step(states[start:end, 0].astype(np.float64), params[start:end])
            next_states[start:end, 0] = np.asarray(stepped, dtype=np.float32)
        except Exception as exc:
            print(
                f"WARNING: PDE step failed for generated chunk {start}:{end}; "
                f"marking it invalid for redraw. Error: {exc}",
                file=sys.stderr,
            )
        if progress_label is not None:
            print(
                f"{progress_label}: solver chunk {start}:{end} "
                f"dt={time.perf_counter() - chunk_t0:.2f}s",
                flush=True,
            )
    return next_states


def safe_simulate_trajectories(
    pde: PDE1D,
    states0: np.ndarray,
    params: np.ndarray,
    steps: int,
    batch_size: int = 256,
    apply_warmup: bool = False,
) -> np.ndarray:
    states0 = np.asarray(states0, dtype=np.float32)
    params = np.asarray(params, dtype=np.float32)
    trajectories = np.full((len(states0), steps + 1, 1, states0.shape[-1]), np.nan, dtype=np.float32)
    for start in range(0, len(states0), batch_size):
        end = min(start + batch_size, len(states0))
        try:
            trajectories[start:end] = pde.simulate(states0[start:end], params[start:end], steps, apply_warmup=apply_warmup)
        except Exception as exc:
            print(
                f"WARNING: PDE trajectory simulation failed for generated chunk {start}:{end}; "
                f"marking it invalid for redraw. Error: {exc}",
                file=sys.stderr,
                flush=True,
            )
    return trajectories


def sample_normalized_params(
    n: int,
    dim: int,
    rng: np.random.Generator,
    prior: str = "uniform",
) -> np.ndarray:
    prior = str(prior or "uniform")
    if prior == "uniform":
        values = rng.random((n, dim))
    elif prior == "edge":
        values = rng.beta(0.5, 0.5, size=(n, dim))
    elif prior == "corner":
        corners = rng.integers(0, 2, size=(n, dim)).astype(np.float32)
        values = corners + rng.normal(0.0, 0.08, size=(n, dim))
    elif prior == "upper":
        values = rng.beta(2.0, 0.5, size=(n, dim))
    elif prior == "lower":
        values = rng.beta(0.5, 2.0, size=(n, dim))
    else:
        raise ValueError(f"Unknown ddpm.param_prior {prior!r}")
    return np.asarray(values, dtype=np.float32).clip(0.0, 1.0)


def propose_conditioning_losses(
    losses: np.ndarray | None,
    n: int,
    rng: np.random.Generator,
    strategy: str = "empirical",
    loss_metric: str = "mse",
    qmin: float = 0.75,
    qmax: float = 1.5,
) -> np.ndarray:
    strategy = str(strategy or "empirical")
    if losses is None or len(losses) == 0:
        return rng.uniform(float(qmin), float(qmax), size=(n, 1)).astype(np.float32)
    if strategy == "empirical":
        return propose_losses(losses, n, rng, metric=loss_metric)

    normalized = normalize_losses(transform_transition_losses(losses, loss_metric)).reshape(-1)
    if strategy == "tail":
        threshold = float(np.quantile(normalized, np.clip(qmin, 0.0, 1.0)))
        candidates = normalized[normalized >= threshold]
        if len(candidates) == 0:
            candidates = normalized
        sampled = rng.choice(candidates, size=n, replace=True)
        sampled = sampled + rng.normal(0.0, 0.03, size=n)
        return np.clip(sampled, 0.0, float(qmax)).reshape(n, 1).astype(np.float32)
    if strategy == "uniform_high":
        lo = float(qmin)
        hi = max(lo + 1e-6, float(qmax))
        return rng.uniform(lo, hi, size=(n, 1)).astype(np.float32)
    if strategy == "max":
        sampled = rng.normal(float(qmax), 0.05, size=n)
        return np.clip(sampled, 0.0, float(qmax)).reshape(n, 1).astype(np.float32)
    if strategy == "quantile_band":
        lo_q = np.clip(float(qmin), 0.0, 1.0)
        hi_q = np.clip(float(qmax), lo_q, 1.0)
        lo, hi = np.quantile(normalized, [lo_q, hi_q])
        candidates = normalized[(normalized >= lo) & (normalized <= hi)]
        if len(candidates) == 0:
            candidates = normalized
        sampled = rng.choice(candidates, size=n, replace=True)
        sampled = sampled + rng.normal(0.0, 0.03, size=n)
        return np.clip(sampled, 0.0, 1.5).reshape(n, 1).astype(np.float32)
    raise ValueError(f"Unknown ddpm.loss_proposal {strategy!r}")


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
    param_prior: str = "uniform",
    sample_temperature: float = 1.0,
    loss_condition_scale: float = 1.0,
    sample_strategy: str = "exp_bias",
    sample_strategy_temp: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate states (physical units) and their PDE parameters from the DDPM.

    Depending on ddpm.param_mode the parameters are either sampled uniformly and used to
    condition the state generation, jointly generated with the state, or sampled uniformly
    and paired afterwards ("none"). The returned params are exactly the ones to feed the
    solver and the surrogate.

    In mode="conditional_quantile" the generator is conditioned on a discrete difficulty
    bin sampled from `sample_strategy` (e.g. exp_bias biases toward hard bins).
    """
    states_out, params_out = [], []
    losses_flat = None if loss_values is None else np.asarray(loss_values, dtype=np.float32).reshape(-1)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = end - start

        valid_states = []
        valid_params = []
        needed = b
        attempts = 0
        max_attempts = 10

        while needed > 0 and attempts < max_attempts:
            attempts += 1
            attempt_t0 = time.perf_counter()
            loss_cond = None
            if mode == "conditional_loss" and loss_values is not None:
                if losses_flat is not None and len(losses_flat) >= n:
                    current_losses = losses_flat[start + (b - needed) : start + (b - needed) + needed]
                elif losses_flat is not None and len(losses_flat) > 0:
                    current_losses = rng.choice(losses_flat, size=needed, replace=True)
                else:
                    current_losses = rng.uniform(0.5, 1.0, size=needed).astype(np.float32)
                current_losses = current_losses * float(loss_condition_scale)
                loss_cond = torch.as_tensor(current_losses, dtype=torch.float32, device=device).view(needed, 1)

            quantile_label = None
            if mode == "conditional_quantile":
                ql = sample_quantile_labels(
                    sample_strategy, needed, ddpm.n_quantiles, rng, sample_strategy_temp
                )
                quantile_label = torch.as_tensor(ql, dtype=torch.long, device=device)

            if ddpm.param_mode == "condition":
                normed = sample_normalized_params(needed, ddpm.param_dim, rng, param_prior)
                param_scaled = torch.from_numpy(2.0 * normed - 1.0).to(device)
                state_norm, _ = ddpm.sample(
                    needed,
                    loss=loss_cond,
                    params=param_scaled,
                    device=device,
                    temperature=sample_temperature,
                    quantile_label=quantile_label,
                )
                params_phys = pde.denormalize_params(normed)
            elif ddpm.param_mode == "generate":
                state_norm, p_scaled = ddpm.sample(
                    needed,
                    loss=loss_cond,
                    params=None,
                    device=device,
                    temperature=sample_temperature,
                    quantile_label=quantile_label,
                )
                normed = ((p_scaled.clamp(-1.0, 1.0).cpu().numpy() + 1.0) / 2.0).astype(np.float32)
                params_phys = pde.denormalize_params(normed)
            else:
                state_norm, _ = ddpm.sample(
                    needed,
                    loss=loss_cond,
                    params=None,
                    device=device,
                    temperature=sample_temperature,
                    quantile_label=quantile_label,
                )
                if param_prior == "uniform":
                    params_phys = pde.sample_params_uniform(needed, rng)
                else:
                    normed = sample_normalized_params(needed, ddpm.param_dim, rng, param_prior)
                    params_phys = pde.denormalize_params(normed)

            state_phys = state_norm.cpu().numpy() * state_std + state_mean

            good_indices = np.where(finite_rows(state_phys, params_phys))[0]
            print(
                f"ddpm generate batch {start}:{end} attempt={attempts} "
                f"requested={needed} finite={len(good_indices)} "
                f"dt={time.perf_counter() - attempt_t0:.2f}s",
                flush=True,
            )
            if len(good_indices) > 0:
                valid_states.append(state_phys[good_indices])
                valid_params.append(params_phys[good_indices])
                needed -= len(good_indices)

        if needed > 0:
            print(
                f"WARNING: DDPM generated NaNs/Infs after {max_attempts} attempts. "
                f"Falling back to uniform sampling for {needed} states.",
                file=sys.stderr,
            )
            fallback_states = pde.sample_ic_uniform(needed, rng)
            fallback_params = pde.sample_params_uniform(needed, rng)
            valid_states.append(fallback_states)
            valid_params.append(fallback_params)

        states_out.append(np.concatenate(valid_states, axis=0)[:b].astype(np.float32))
        params_out.append(np.concatenate(valid_params, axis=0)[:b].astype(np.float32))

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
    loss_proposal: str = "empirical",
    loss_metric: str = "mse",
    loss_quantile_min: float = 0.75,
    loss_quantile_max: float = 1.5,
    param_prior: str = "uniform",
    sample_temperature: float = 1.0,
    loss_condition_scale: float = 1.0,
    sample_batch_size: int = 512,
    generated_pool_mode: str = "trajectory",
    solver_batch_size: int = 256,
    sample_strategy: str = "exp_bias",
    sample_strategy_temp: float = 0.5,
) -> TransitionPool:
    n_uniform = int(round(n_traj * uniform_fraction))
    n_generated_traj = n_traj - n_uniform
    pools = []
    if n_uniform > 0:
        pools.append(make_uniform_pool(pde, n_uniform, steps, rng))
    generated_pool_mode = str(generated_pool_mode or "trajectory")
    if n_generated_traj > 0 and generated_pool_mode == "trajectory":
        needed = n_generated_traj
        valid_traj = []
        valid_params = []
        valid_proposed_losses = []
        attempts = 0
        max_attempts = 10

        while needed > 0 and attempts < max_attempts:
            attempts += 1
            current_loss_values = propose_conditioning_losses(
                previous.losses,
                needed,
                rng,
                strategy=loss_proposal,
                loss_metric=loss_metric,
                qmin=loss_quantile_min,
                qmax=loss_quantile_max,
            )
            states_attempt, params_attempt = generate_states_and_params(
                ddpm,
                needed,
                ddpm_mode,
                current_loss_values if ddpm_mode == "conditional_loss" else None,
                pde,
                state_mean,
                state_std,
                rng,
                device,
                batch_size=sample_batch_size,
                param_prior=param_prior,
                sample_temperature=sample_temperature,
                loss_condition_scale=loss_condition_scale,
                sample_strategy=sample_strategy,
                sample_strategy_temp=sample_strategy_temp,
            )
            traj_attempt = safe_simulate_trajectories(
                pde,
                states_attempt,
                params_attempt,
                steps,
                batch_size=solver_batch_size,
            )
            good_indices = np.where(finite_rows(traj_attempt, params_attempt))[0]
            if len(good_indices) > 0:
                valid_traj.append(traj_attempt[good_indices])
                valid_params.append(params_attempt[good_indices])
                valid_proposed_losses.append(current_loss_values[good_indices])
                needed -= len(good_indices)

        if needed > 0:
            print(
                f"WARNING: DDPM generated trajectories produced NaNs/Infs after {max_attempts} "
                f"attempts. Falling back to uniform sampling for {needed} trajectories.",
                file=sys.stderr,
                flush=True,
            )
            fallback_pool = make_uniform_pool(pde, needed, steps, rng)
            traj = np.empty((needed, steps + 1, 1, pde.resolution), dtype=np.float32)
            traj[:, 0] = fallback_pool.states.reshape(needed, steps, 1, pde.resolution)[:, 0]
            traj[:, 1:] = fallback_pool.next_states.reshape(needed, steps, 1, pde.resolution)
            fallback_params = fallback_pool.params.reshape(needed, steps, -1)[:, 0]
            fallback_loss_values = propose_conditioning_losses(
                previous.losses,
                needed,
                rng,
                strategy=loss_proposal,
                loss_metric=loss_metric,
                qmin=loss_quantile_min,
                qmax=loss_quantile_max,
            )
            valid_traj.append(traj)
            valid_params.append(fallback_params)
            valid_proposed_losses.append(fallback_loss_values)

        traj = np.concatenate(valid_traj, axis=0)[:n_generated_traj].astype(np.float32)
        params = np.concatenate(valid_params, axis=0)[:n_generated_traj].astype(np.float32)
        loss_values = np.concatenate(valid_proposed_losses, axis=0)[:n_generated_traj].astype(np.float32)
        generated_pool = TransitionPool.from_trajectories(traj, params)
        generated_pool.source = np.ones((len(generated_pool),), dtype=np.int8)
        generated_pool.proposed_losses = np.repeat(loss_values.reshape(-1), steps).astype(np.float32)
        pools.append(generated_pool)
    elif n_generated_traj > 0 and generated_pool_mode == "transition":
        n_generated = n_generated_traj * steps
        
        needed = n_generated
        valid_states0 = []
        valid_params = []
        valid_next_states = []
        valid_proposed_losses = []
        attempts = 0
        max_attempts = 10
        
        while needed > 0 and attempts < max_attempts:
            attempts += 1
            current_loss_values = propose_conditioning_losses(
                previous.losses,
                needed,
                rng,
                strategy=loss_proposal,
                loss_metric=loss_metric,
                qmin=loss_quantile_min,
                qmax=loss_quantile_max,
            )
                
            states_attempt, params_attempt = generate_states_and_params(
                ddpm,
                needed,
                ddpm_mode,
                current_loss_values if ddpm_mode == "conditional_loss" else None,
                pde,
                state_mean,
                state_std,
                rng,
                device,
                batch_size=sample_batch_size,
                param_prior=param_prior,
                sample_temperature=sample_temperature,
                loss_condition_scale=loss_condition_scale,
                sample_strategy=sample_strategy,
                sample_strategy_temp=sample_strategy_temp,
            )
            
            next_state_attempt = safe_step_transitions(
                pde,
                states_attempt,
                params_attempt,
                batch_size=solver_batch_size,
                progress_label=f"mixed round transition attempt={attempts}",
            )

            good_indices = np.where(finite_rows(states_attempt, params_attempt, next_state_attempt))[0]
            print(
                f"mixed pool transition attempt={attempts} requested={needed} "
                f"valid_after_solver={len(good_indices)}",
                flush=True,
            )
            if len(good_indices) > 0:
                valid_states0.append(states_attempt[good_indices])
                valid_params.append(params_attempt[good_indices])
                valid_next_states.append(next_state_attempt[good_indices])
                valid_proposed_losses.append(current_loss_values[good_indices])
                needed -= len(good_indices)

        if needed > 0:
            print(
                f"WARNING: DDPM generation or transition simulation produced NaNs/Infs "
                f"after {max_attempts} attempts. Falling back to uniform sampling for "
                f"{needed} transitions.",
                file=sys.stderr,
            )
            fallback_pool = make_uniform_pool(pde, needed, 1, rng)
            fallback_states = fallback_pool.states
            fallback_params = fallback_pool.params
            fallback_next_states = fallback_pool.next_states
            fallback_loss_values = propose_conditioning_losses(
                previous.losses,
                needed,
                rng,
                strategy=loss_proposal,
                loss_metric=loss_metric,
                qmin=loss_quantile_min,
                qmax=loss_quantile_max,
            )

            valid_states0.append(fallback_states)
            valid_params.append(fallback_params)
            valid_next_states.append(fallback_next_states)
            valid_proposed_losses.append(fallback_loss_values)

        states0 = np.concatenate(valid_states0, axis=0)[:n_generated].astype(np.float32)
        params = np.concatenate(valid_params, axis=0)[:n_generated].astype(np.float32)
        next_state = np.concatenate(valid_next_states, axis=0)[:n_generated].astype(np.float32)
        loss_values = np.concatenate(valid_proposed_losses, axis=0)[:n_generated].astype(np.float32)

        pools.append(
            TransitionPool(
                states0,
                params,
                next_state,
                source=np.ones((n_generated,), dtype=np.int8),
                proposed_losses=loss_values.reshape(-1).astype(np.float32),
            )
        )
    elif n_generated_traj > 0:
        raise ValueError(f"Unknown ddpm.generated_pool_mode {generated_pool_mode!r}")
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
