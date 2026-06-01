from __future__ import annotations

import json
import os
import random
import sys
import argparse
import io
import time
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
from .generator_eval import evaluate_generator
from .models.ddpm import DDPM1D, FlowMatching1D
from .models.surrogate import build_surrogate
from .pde import PDE1D
from .train import (
    compute_transition_losses,
    normalize_losses,
    propose_losses,
    sample_quantile_labels,
    train_ddpm,
    train_surrogate,
    transform_transition_losses,
)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


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
    ):
        run.define_metric(f"{namespace}/*", step_metric="round")

    return run



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


def metric_name(prefix: str, suffix: str) -> str:
    if "/" in prefix:
        namespace, base = prefix.split("/", 1)
        return f"{namespace}/{base}_{suffix}"
    return f"{prefix}/{suffix}"


def state_quality_metrics(states: np.ndarray, prefix: str) -> dict[str, float]:
    flat = np.asarray(states, dtype=np.float32)
    finite = np.isfinite(flat)
    metrics = {metric_name(prefix, "finite_ratio"): float(finite.mean())}
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
            metric_name(prefix, "value_std"): float(clean.std()),
            metric_name(prefix, "abs_p99"): float(np.quantile(np.abs(clean), 0.99)),
            metric_name(prefix, "total_variation_mean"): float(np.mean(np.sum(np.abs(dx), axis=-1))),
            metric_name(prefix, "roughness_mean"): float(np.mean(dx**2)),
            metric_name(prefix, "highfreq_energy_ratio_mean"): float(np.mean(high_power / total_power)),
        }
    )
    return metrics


def pool_debug_metrics(
    pool: TransitionPool,
    pretrain_losses: np.ndarray | None = None,
    loss_metric: str = "mse",
) -> dict[str, float]:
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
        if name == "generated":
            div = state_diversity(states)
            metrics[f"pool/{name}_pairwise_l2_mean"] = div["pairwise_l2_mean"]
            metrics[f"pool/{name}_pairwise_l2_p10"] = div["pairwise_l2_p10"]
            metrics[f"pool/{name}_pairwise_l2_p90"] = div["pairwise_l2_p90"]
            metrics.update(state_quality_metrics(states, f"pool/{name}_states"))
        if pretrain_losses is not None:
            losses = pretrain_losses[mask]
            metrics[f"pool/{name}_pretrain_loss_mean"] = float(losses.mean())
            metrics[f"pool/{name}_pretrain_loss_p90"] = float(np.quantile(losses, 0.9))
            if name == "generated":
                metrics[f"pool/{name}_pretrain_loss_p99"] = float(np.quantile(losses, 0.99))
        if pool.losses is not None:
            losses = pool.losses[mask]
            metrics[f"pool/{name}_posttrain_loss_mean"] = float(losses.mean())
            metrics[f"pool/{name}_posttrain_loss_p90"] = float(np.quantile(losses, 0.9))
            if name == "generated":
                metrics[f"pool/{name}_posttrain_loss_p99"] = float(np.quantile(losses, 0.99))
    if (
        pretrain_losses is not None
        and pool.proposed_losses is not None
        and np.isfinite(pool.proposed_losses).any()
    ):
        gen = source == 1
        proposed = pool.proposed_losses[gen]
        proposed = proposed[np.isfinite(proposed)]
        if len(proposed) > 0:
            metrics["pool/generated_proposed_loss_mean"] = float(proposed.mean())
            metrics["pool/generated_proposed_loss_p90"] = float(np.quantile(proposed, 0.9))
            metrics["pool/generated_proposed_loss_p99"] = float(np.quantile(proposed, 0.99))
        pretrain_condition_losses = normalize_losses(
            transform_transition_losses(pretrain_losses[gen], loss_metric)
        )
        metrics["pool/generated_proposed_vs_pretrain_loss_corr"] = safe_corr(
            pool.proposed_losses[gen], pretrain_condition_losses
        )
    if pool.losses is not None and pool.source is not None:
        gen = source == 1
        if gen.any() and pool.proposed_losses is not None:
            posttrain_condition_losses = normalize_losses(
                transform_transition_losses(pool.losses[gen], loss_metric)
            )
            metrics["pool/generated_proposed_vs_posttrain_loss_corr"] = safe_corr(
                pool.proposed_losses[gen], posttrain_condition_losses
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


def heatmap_image(
    values: np.ndarray,
    cmap: str = "coolwarm",
    center_zero: bool = True,
    scale: float | None = None,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if center_zero:
        signed_scale = float(scale) if scale is not None else float(np.quantile(np.abs(values), 0.98))
        signed_scale = max(signed_scale, 1e-8)
        scaled = 0.5 + 0.5 * (values / signed_scale)
    else:
        if value_range is None:
            lo, hi = np.quantile(values, [0.02, 0.98])
        else:
            lo, hi = value_range
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


def rollout_error_pyplot_image(series: dict[str, np.ndarray], metric: str) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    step = series["step"]
    for key, linestyle in (("mean", "-"), ("p95", "--"), ("p99", ":")):
        values = series[f"{metric}_{key}"]
        ax.plot(step, values, linestyle=linestyle, linewidth=2.0, label=key)
    ax.set_title(f"Rollout {metric.upper()} per step")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def spacetime_grid_image(spacetime: dict[str, np.ndarray], n_rows: int = 3) -> np.ndarray:
    """Build a single (n_rows x 3) space-time grid image: each row is one example,
    columns are [ground truth | prediction | absolute error]. Per-row color scale is
    anchored to that example's ground-truth q98 amplitude. Rows are separated by borders.
    """
    n_rows = int(min(n_rows, spacetime["truth"].shape[0]))
    rows = []
    row_w = None
    for i in range(n_rows):
        gt_scale = max(float(np.quantile(np.abs(spacetime["truth"][i]), 0.98)), 1e-8)
        gt_rgb = heatmap_image(spacetime["truth"][i], cmap="coolwarm", center_zero=True, scale=gt_scale)
        H, W, _ = gt_rgb.shape
        indices = np.round(np.linspace(0, W - 1, H)).astype(np.int64)
        gt_sq = gt_rgb[:, indices, :]
        pred_sq = heatmap_image(
            spacetime["prediction"][i], cmap="coolwarm", center_zero=True, scale=gt_scale
        )[:, indices, :]
        abs_err_sq = heatmap_image(
            np.abs(spacetime["error"][i]), cmap="magma", center_zero=False, value_range=(0.0, gt_scale)
        )[:, indices, :]
        bw = max(2, H // 100)
        vborder = np.zeros((H, bw, 3), dtype=np.uint8)
        row = np.concatenate([gt_sq, vborder, pred_sq, vborder, abs_err_sq], axis=1)
        rows.append(row)
        row_w = row.shape[1]
    hborder = np.zeros((max(2, rows[0].shape[0] // 100), row_w, 3), dtype=np.uint8)
    stacked = []
    for i, r in enumerate(rows):
        if i > 0:
            stacked.append(hborder)
        stacked.append(r)
    return np.concatenate(stacked, axis=0)


def state_profile_grid_image(
    states: np.ndarray,
    max_examples: int = 8,
    width: int = 900,
    row_height: int = 112,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 3:
        states = states[:, 0]
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    n = min(max_examples, len(states))
    if n == 0:
        return np.full((row_height, width, 3), 255, dtype=np.uint8)

    idx = np.linspace(0, len(states) - 1, n, dtype=np.int64)
    profiles = states[idx]
    scale = float(np.quantile(np.abs(profiles), 0.98))
    scale = max(scale, 1e-8)

    pad_x = 18
    pad_y = 12
    height = n * row_height
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x = np.linspace(pad_x, width - pad_x - 1, profiles.shape[1])

    def draw_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]):
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, steps).round().astype(np.int64)
        ys = np.linspace(y0, y1, steps).round().astype(np.int64)
        ok = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
        img[ys[ok], xs[ok]] = color

    line_colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
    ]
    for row, profile in enumerate(profiles):
        top = row * row_height
        bottom = top + row_height - 1
        center = top + row_height // 2
        usable = row_height - 2 * pad_y
        y = center - np.clip(profile / scale, -1.0, 1.0) * (usable / 2.0)

        image[top:bottom, pad_x] = (210, 210, 210)
        image[top:bottom, width - pad_x - 1] = (210, 210, 210)
        image[center, pad_x : width - pad_x] = (220, 220, 220)
        image[top + pad_y, pad_x : width - pad_x] = (235, 235, 235)
        image[bottom - pad_y, pad_x : width - pad_x] = (235, 235, 235)

        color = line_colors[row % len(line_colors)]
        for k in range(len(x) - 1):
            draw_line(image, x[k], y[k], x[k + 1], y[k + 1], color)
    return image


def state_profile_line_series(states: np.ndarray, max_examples: int = 8, max_points: int = 400):
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 3:
        states = states[:, 0]
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    n = min(max_examples, len(states))
    if n == 0:
        return [0], [[0.0]], ["empty"]
    sample_idx = np.linspace(0, len(states) - 1, n, dtype=np.int64)
    grid_idx = np.linspace(0, states.shape[1] - 1, min(max_points, states.shape[1]), dtype=np.int64)
    xs = grid_idx.tolist()
    ys = [states[i, grid_idx].astype(float).tolist() for i in sample_idx]
    keys = [f"sample_{j}" for j in range(n)]
    return xs, ys, keys


def epoch_validation_count(cfg, validation) -> int:
    ten_percent = max(1, int(np.ceil(0.10 * validation.n_trajectories)))
    configured = int(getattr(cfg.validation, "epoch_n_trajectories", ten_percent) or ten_percent)
    return max(1, min(configured, ten_percent, validation.n_trajectories))


def wandb_group_label(cfg) -> str:
    label = str(cfg.wandb.group or "default")
    for prefix in ("ks800_5case_", "ks800_al4pde_exact_5seed_viz_", "ks800_ready_"):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


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

    seed_everything(cfg.seed)
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

    # Real-KS reference states (skip t=0) for the per-round generator realism metrics.
    _val_traj = validation.trajectories
    _val_states_flat = _val_traj[:, 1:].reshape(-1, 1, _val_traj.shape[-1]).astype(np.float32)
    _gen_eval_idx = rng.choice(len(_val_states_flat), size=min(1024, len(_val_states_flat)), replace=False)
    val_ref_states = _val_states_flat[_gen_eval_idx]

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
    generator_cls = FlowMatching1D if str(cfg.ddpm.generator).lower() in ("flow", "flow_matching", "fm") else DDPM1D
    ddpm = generator_cls(
        resolution=cfg.pde.resolution,
        hidden=cfg.ddpm.hidden,
        steps=cfg.ddpm.steps,
        loss_conditional=cfg.ddpm.mode == "conditional_loss",
        param_dim=len(pde.param_ranges),
        param_mode=cfg.ddpm.param_mode,
        residual_blocks=cfg.ddpm.residual_blocks,
        kernel_size=cfg.ddpm.kernel_size,
        n_quantiles=cfg.ddpm.n_quantiles if cfg.ddpm.mode == "conditional_quantile" else 0,
        quant_embed_dim=cfg.ddpm.quant_embed_dim,
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
        round_start_time = time.perf_counter()
        timing_metrics: dict[str, float] = {}

        def finish_phase(name: str, started_at: float) -> None:
            elapsed = time.perf_counter() - started_at
            timing_metrics[f"timing/{name}_sec"] = float(elapsed)
            print(f"round {round_id}: timing {name}={elapsed:.2f}s", flush=True)

        if round_id > 0:
            print(f"round {round_id}: generating mixed pool", flush=True)
            phase_start = time.perf_counter()
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
                loss_proposal=cfg.ddpm.loss_proposal,
                loss_metric=cfg.ddpm.loss_metric,
                loss_quantile_min=cfg.ddpm.loss_quantile_min,
                loss_quantile_max=cfg.ddpm.loss_quantile_max,
                param_prior=cfg.ddpm.param_prior,
                sample_temperature=cfg.ddpm.sample_temperature,
                loss_condition_scale=cfg.ddpm.loss_condition_scale,
                sample_batch_size=cfg.ddpm.sample_batch_size,
                generated_pool_mode=cfg.ddpm.generated_pool_mode,
                solver_batch_size=cfg.ddpm.solver_batch_size,
                sample_strategy=cfg.ddpm.sample_strategy,
                sample_strategy_temp=cfg.ddpm.sample_strategy_temp,
            )
            finish_phase("make_pool", phase_start)
        else:
            print("round 0: using initial uniform pool", flush=True)
        print(f"round {round_id}: computing pretrain losses", flush=True)
        phase_start = time.perf_counter()
        pretrain_losses = compute_transition_losses(model, pool, cfg.surrogate.batch_size, device)
        pool.pretrain_losses = pretrain_losses
        finish_phase("pretrain_losses", phase_start)
        print(f"round {round_id}: training surrogate", flush=True)
        phase_start = time.perf_counter()
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
            seed=cfg.seed + round_id,
        )
        finish_phase("surrogate_train", phase_start)
        ddpm_metrics = {}
        if cfg.ddpm.enabled:
            print(f"round {round_id}: training ddpm", flush=True)
            phase_start = time.perf_counter()
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
                loss_metric=cfg.ddpm.loss_metric,
                difficulty_signal=cfg.ddpm.difficulty_signal,
                param_loss_weight=cfg.ddpm.param_loss_weight,
                loss_condition_scale=cfg.ddpm.loss_condition_scale,
                sample_weight_floor=cfg.ddpm.sample_weight_floor,
                sample_weight_power=cfg.ddpm.sample_weight_power,
                epoch_callback=make_epoch_callback(
                    run, model, validation, cfg, device, round_id, "ddpm_epoch"
                ),
                round_id=round_id,
            )
            finish_phase("generator_train", phase_start)
        gen_eval_metrics: dict[str, float] = {}
        if cfg.ddpm.enabled and cfg.ddpm.mode == "conditional_quantile":
            print(f"round {round_id}: evaluating generator", flush=True)
            phase_start = time.perf_counter()
            ref_loss_mean = float(np.mean(pool.losses)) if pool.losses is not None else 1.0
            gen_eval_metrics = evaluate_generator(
                ddpm=ddpm,
                surrogate=model,
                pde=pde,
                val_ref_states=val_ref_states,
                ref_loss_mean=ref_loss_mean,
                state_mean=state_mean,
                state_std=state_std,
                rng=rng,
                device=device,
                loss_metric=cfg.ddpm.loss_metric,
                difficulty_signal=cfg.ddpm.difficulty_signal,
                n_eval=min(512, cfg.ddpm.sample_batch_size),
                solver_batch_size=cfg.ddpm.solver_batch_size,
                surrogate_batch_size=cfg.surrogate.batch_size,
            )
            finish_phase("generator_eval", phase_start)
            if gen_eval_metrics:
                print(json.dumps({k: round(v, 4) for k, v in gen_eval_metrics.items()}, sort_keys=True), flush=True)
        print(f"round {round_id}: full validation", flush=True)
        phase_start = time.perf_counter()
        eval_metrics = evaluate_validation(
            model=model,
            validation=validation,
            rollout_steps=cfg.validation.rollout_steps,
            quantiles=cfg.validation.quantiles,
            device=device,
        )
        finish_phase("full_validation", phase_start)
        timing_metrics["timing/round_before_diagnostics_sec"] = float(time.perf_counter() - round_start_time)
        metrics = {
            "round": round_id,
            "pool/n_samples": len(pool),
            "pool/loss_mean": float(np.mean(pool.losses)) if pool.losses is not None else 0.0,
            "pool/loss_p90": float(np.quantile(pool.losses, 0.9)) if pool.losses is not None else 0.0,
            **train_metrics,
            **ddpm_metrics,
            **gen_eval_metrics,
            **eval_metrics,
            **timing_metrics,
        }
        metrics.update(pool_debug_metrics(pool, pretrain_losses, cfg.ddpm.loss_metric))
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if run is not None:
            run.log(metrics)
            try:
                phase_start = time.perf_counter()
                log_wandb_diagnostics(
                    run, model, ddpm, pool, validation, cfg, device, round_id, round_id,
                    pde, state_mean, state_std,
                )
                diagnostics_sec = time.perf_counter() - phase_start
                print(f"round {round_id}: timing diagnostics={diagnostics_sec:.2f}s", flush=True)
                run.log({"round": round_id, "timing/diagnostics_sec": float(diagnostics_sec)})
            except Exception as exc:
                print(f"wandb diagnostics failed at round {round_id}: {exc}", file=sys.stderr)
        phase_start = time.perf_counter()
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
        save_sec = time.perf_counter() - phase_start
        round_total_sec = time.perf_counter() - round_start_time
        print(
            f"round {round_id}: timing save={save_sec:.2f}s total={round_total_sec:.2f}s",
            flush=True,
        )
        if run is not None:
            run.log(
                {
                    "round": round_id,
                    "timing/save_sec": float(save_sec),
                    "timing/round_total_sec": float(round_total_sec),
                }
            )

    (out / "history.json").write_text(json.dumps(history, indent=2))
    if run is not None:
        import os
        import signal
        import threading

        finished = threading.Event()

        def _finish():
            try:
                run.finish()
            finally:
                finished.set()

        t = threading.Thread(target=_finish, daemon=True)
        t.start()
        # Give wandb up to 3 minutes to upload; if it hangs, force-exit so the
        # OAR allocation is released rather than sitting idle on the GPU node.
        if not finished.wait(timeout=180):
            print("wandb.finish() timed out after 180s — forcing exit.", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
