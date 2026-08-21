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


def state_total_variation(states: np.ndarray) -> np.ndarray:
    """Per-state periodic total variation (physical units), shape [N]."""
    s = np.asarray(states, dtype=np.float32)[:, 0]
    tv = np.abs(np.diff(s, axis=-1)).sum(axis=-1) + np.abs(s[:, 0] - s[:, -1])
    return tv.astype(np.float32)


@torch.no_grad()
def ensemble_disagreement(surrogate, states: np.ndarray, params: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    """Per-state ensemble disagreement (forward passes only; zero solver cost)."""
    surrogate.eval()
    out = []
    for i in range(0, len(states), batch_size):
        s = torch.from_numpy(np.asarray(states[i : i + batch_size], dtype=np.float32)).to(device)
        p = torch.from_numpy(np.asarray(params[i : i + batch_size], dtype=np.float32)).to(device)
        out.append(surrogate.uncertainty(s, p).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


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
    surrogate=None,
    candidate_factor: int = 1,
    tv_gate_threshold: float = 0.0,
    anchors_phys: np.ndarray | None = None,
    sample_mode: str = "scratch",
    edit_t0: float = 0.6,
    previous: TransitionPool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate states (physical units), their PDE parameters, and the requested
    difficulty bin per kept state (-1 where not applicable).

    Steering happens in two stages:
      1. conditioning — quantile-bin label (mode="conditional_quantile") and, in
         sample_mode="edit", SDEdit-style generation around real attractor anchors;
      2. selection — with candidate_factor > 1 and a surrogate ensemble, generate
         candidate_factor x more candidates, drop the unphysical ones (TV gate),
         and keep the states with the highest ensemble disagreement. Selection by
         REALIZED disagreement replaces weak conditioning as the steering mechanism
         and costs zero extra solver calls.
    """
    candidate_factor = max(1, int(candidate_factor))
    use_edit = str(sample_mode) == "edit" and anchors_phys is not None and len(anchors_phys) > 0
    anchors_norm = None
    if use_edit:
        anchors_norm = ((np.asarray(anchors_phys, dtype=np.float32) - state_mean) / state_std).astype(np.float32)

    states_out, params_out, bins_out = [], [], []
    losses_flat = None if loss_values is None else np.asarray(loss_values, dtype=np.float32).reshape(-1)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = end - start

        valid_states, valid_params, valid_bins = [], [], []
        needed = b
        attempts = 0
        max_attempts = 10

        while needed > 0 and attempts < max_attempts:
            attempts += 1
            attempt_t0 = time.perf_counter()
            want = needed * candidate_factor

            loss_cond = None
            if mode == "conditional_loss" and loss_values is not None:
                if losses_flat is not None and len(losses_flat) > 0:
                    current_losses = rng.choice(losses_flat, size=want, replace=True)
                else:
                    current_losses = rng.uniform(0.5, 1.0, size=want).astype(np.float32)
                current_losses = current_losses * float(loss_condition_scale)
                loss_cond = torch.as_tensor(current_losses, dtype=torch.float32, device=device).view(want, 1)

            quantile_label = None
            ql = np.full(want, -1, dtype=np.int64)
            if mode == "conditional_quantile":
                ql = sample_quantile_labels(
                    sample_strategy, want, ddpm.n_quantiles, rng, sample_strategy_temp
                )
                quantile_label = torch.as_tensor(ql, dtype=torch.long, device=device)

            edit_kwargs = {}
            if use_edit:
                anchor_idx = rng.choice(len(anchors_norm), size=want, replace=len(anchors_norm) < want)
                edit_kwargs = {
                    "init_state": torch.from_numpy(anchors_norm[anchor_idx]).to(device),
                    "t_start": float(edit_t0),
                }

            amplitude_cond = None
            if getattr(ddpm, "amplitude_conditional", False) and previous is not None:
                prev_amps = np.sqrt(np.mean(previous.next_states**2, axis=(1, 2)))
                # Propose target amplitudes: sample uniformly between min and max of previous pool target amplitudes
                # scaled by state_std
                amps = rng.uniform(prev_amps.min(), prev_amps.max(), size=want).astype(np.float32)
                amplitude_cond = torch.from_numpy(amps / state_std).to(device)

            if ddpm.param_mode == "condition":
                normed = sample_normalized_params(want, ddpm.param_dim, rng, param_prior)
                param_scaled = torch.from_numpy(2.0 * normed - 1.0).to(device)
                state_norm, _ = ddpm.sample(
                    want,
                    loss=loss_cond,
                    params=param_scaled,
                    device=device,
                    temperature=sample_temperature,
                    quantile_label=quantile_label,
                    amplitude=amplitude_cond,
                    **edit_kwargs,
                )
                params_phys = pde.denormalize_params(normed)
            elif ddpm.param_mode == "generate":
                state_norm, p_scaled = ddpm.sample(
                    want,
                    loss=loss_cond,
                    params=None,
                    device=device,
                    temperature=sample_temperature,
                    quantile_label=quantile_label,
                    amplitude=amplitude_cond,
                    **edit_kwargs,
                )
                normed = ((p_scaled.clamp(-1.0, 1.0).cpu().numpy() + 1.0) / 2.0).astype(np.float32)
                params_phys = pde.denormalize_params(normed)
            else:
                state_norm, _ = ddpm.sample(
                    want,
                    loss=loss_cond,
                    params=None,
                    device=device,
                    temperature=sample_temperature,
                    quantile_label=quantile_label,
                    amplitude=amplitude_cond,
                    **edit_kwargs,
                )
                if param_prior == "uniform":
                    params_phys = pde.sample_params_uniform(want, rng)
                else:
                    normed = sample_normalized_params(want, ddpm.param_dim, rng, param_prior)
                    params_phys = pde.denormalize_params(normed)

            state_phys = state_norm.cpu().numpy() * state_std + state_mean

            keep = finite_rows(state_phys, params_phys)
            n_finite = int(keep.sum())
            # Realism gate: drop candidates rougher than the physical reference.
            n_gated = 0
            if tv_gate_threshold > 0:
                tv = state_total_variation(state_phys)
                gate = keep & (tv <= float(tv_gate_threshold))
                # never gate below what we need — relax to finite-only if too strict
                if int(gate.sum()) >= min(needed, n_finite):
                    n_gated = n_finite - int(gate.sum())
                    keep = gate
            idx = np.where(keep)[0]
            # Selection: rank surviving candidates by ensemble disagreement.
            if candidate_factor > 1 and surrogate is not None and len(idx) > needed:
                dis = ensemble_disagreement(surrogate, state_phys[idx], params_phys[idx], device)
                idx = idx[np.argsort(-dis)[:needed]]
            else:
                idx = idx[:needed]
            print(
                f"ddpm generate batch {start}:{end} attempt={attempts} "
                f"requested={needed} candidates={want} finite={n_finite} gated_out={n_gated} kept={len(idx)} "
                f"dt={time.perf_counter() - attempt_t0:.2f}s",
                flush=True,
            )
            if len(idx) > 0:
                valid_states.append(state_phys[idx])
                valid_params.append(params_phys[idx])
                valid_bins.append(ql[idx])
                needed -= len(idx)

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
            valid_bins.append(np.full(needed, -1, dtype=np.int64))

        states_out.append(np.concatenate(valid_states, axis=0)[:b].astype(np.float32))
        params_out.append(np.concatenate(valid_params, axis=0)[:b].astype(np.float32))
        bins_out.append(np.concatenate(valid_bins, axis=0)[:b])

    return (
        np.concatenate(states_out, axis=0),
        np.concatenate(params_out, axis=0),
        np.concatenate(bins_out, axis=0).astype(np.int64),
    )


def _tube_perturbation(anchors: np.ndarray, rng: np.random.Generator, rho_min: float, rho_max: float, kmax: int) -> np.ndarray:
    """Random low-wavenumber perturbations of real anchor states (the no-learning
    tube baseline): k in 1..kmax with 1/k amplitudes, mean-zero, unit-RMS, scaled
    per-state by rho * RMS(u) with rho ~ U[rho_min, rho_max]."""
    anchors = np.asarray(anchors, dtype=np.float32)
    n, _, L = anchors.shape
    x = np.arange(L, dtype=np.float64) / L
    pert = np.zeros((n, L), dtype=np.float64)
    for k in range(1, int(kmax) + 1):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
        pert += (1.0 / k) * np.cos(2.0 * np.pi * k * x[None, :] + phases)
    pert -= pert.mean(axis=1, keepdims=True)
    pert /= np.sqrt((pert**2).mean(axis=1, keepdims=True)) + 1e-12
    rho = rng.uniform(float(rho_min), float(rho_max), size=(n, 1))
    rms_u = np.sqrt((anchors[:, 0].astype(np.float64) ** 2).mean(axis=1, keepdims=True))
    out = anchors[:, 0] + (rho * np.maximum(rms_u, 1e-6) * pert).astype(np.float32)
    return out[:, None, :].astype(np.float32)


def _anchor_states(pools: list, previous: TransitionPool | None, rng: np.random.Generator, max_n: int = 4096) -> np.ndarray:
    """Physically-grounded anchor states: prefer this round's fresh uniform half,
    else the previous pool's uniform-sourced states, else the previous pool."""
    cand = None
    if pools:
        cand = pools[0].states
    elif previous is not None:
        if previous.source is not None and (np.asarray(previous.source) == 0).any():
            cand = previous.states[np.asarray(previous.source) == 0]
        else:
            cand = previous.states
    if cand is None or len(cand) == 0:
        raise ValueError("No anchor states available (empty uniform half and previous pool)")
    idx = rng.choice(len(cand), size=min(max_n, len(cand)), replace=False)
    return np.asarray(cand[idx], dtype=np.float32)


@torch.no_grad()
def _keep_hardest(states, params, surrogate, n, device):
    """Garde les n candidats les plus durs = ceux sur lesquels l'ensemble du surrogate
    est le plus en désaccord. Sans surrogate (round 0), garde simplement les n premiers."""
    if surrogate is not None:
        dis = ensemble_disagreement(surrogate, states, params, device)
        idx = np.argsort(-dis)[:n]
    else:
        idx = np.arange(n)
    return states[idx].astype(np.float32), params[idx].astype(np.float32)


def make_strategy_transitions(
    strategy: str,
    pde: PDE1D,
    surrogate,
    anchors: np.ndarray,
    n: int,
    rng: np.random.Generator,
    device: torch.device,
    solver_batch_size: int,
    tube_rho_min: float,
    tube_rho_max: float,
    tube_kmax: int,
    candidate_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Baselines SANS générateur pour la moitié non-uniforme (1 pas de solveur par état gardé).

    random_tube    : ancres + perturbations basse-fréquence aléatoires (ni apprentissage, ni sélection).
    tube_select    : random_tube (SOURCE) + sélection par désaccord — oversample x candidate_factor puis
                     garde les plus durs. Ablation qui isole la valeur de la sélection sur des
                     perturbations hors-attracteur.
    mined_ic       : candidate_factor x IC aléatoires, garde les plus durs (sélection sans génération).
    uniform_select : ancres uniformes ON-ATTRACTOR + sélection par désaccord — isole la valeur du
                     modèle génératif par rapport au simple tri actif de données d'attracteur.
    """
    if strategy == "random_tube":
        anchor_idx = rng.choice(len(anchors), size=n, replace=len(anchors) < n)
        states = _tube_perturbation(anchors[anchor_idx], rng, tube_rho_min, tube_rho_max, tube_kmax)
        return states, pde.sample_params_uniform(n, rng)
    want = n * max(2, int(candidate_factor))  # on sur-échantillonne puis on garde les n plus durs
    if strategy == "tube_select":
        anchor_idx = rng.choice(len(anchors), size=want, replace=len(anchors) < want)
        cand = _tube_perturbation(anchors[anchor_idx], rng, tube_rho_min, tube_rho_max, tube_kmax)
        return _keep_hardest(cand, pde.sample_params_uniform(want, rng), surrogate, n, device)
    if strategy == "mined_ic":
        cand = pde.sample_ic_uniform(want, rng)
        return _keep_hardest(cand, pde.sample_params_uniform(want, rng), surrogate, n, device)
    if strategy == "uniform_select":
        # On-attractor active selection: oversample anchors (real attractor states),
        # rank by ensemble disagreement, keep the N most uncertain.
        anchor_idx = rng.choice(len(anchors), size=want, replace=len(anchors) < want)
        cand = anchors[anchor_idx].astype(np.float32)
        return _keep_hardest(cand, pde.sample_params_uniform(want, rng), surrogate, n, device)
    raise ValueError(f"pool.strategy inconnue : {strategy!r}")


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
    surrogate=None,
    strategy: str = "generator",
    tube_rho_min: float = 0.05,
    tube_rho_max: float = 0.5,
    tube_kmax: int = 12,
    sample_mode: str = "scratch",
    edit_t0: float = 0.6,
    candidate_factor: int = 1,
    realism_tv_gate: float = 0.0,
) -> TransitionPool:
    n_uniform = int(round(n_traj * uniform_fraction))
    n_generated_traj = n_traj - n_uniform
    pools = []
    if n_uniform > 0:
        pools.append(make_uniform_pool(pde, n_uniform, steps, rng))

    # Anchors + realism reference from physically-grounded states only.
    anchors = None
    tv_gate_threshold = 0.0
    if n_generated_traj > 0:
        anchors = _anchor_states(pools, previous, rng)
        if realism_tv_gate > 0:
            tv_gate_threshold = float(realism_tv_gate) * float(np.quantile(state_total_variation(anchors), 0.99))

    # Generator-free baseline strategies fill the non-uniform half with single-step
    # transitions at exactly one solver step per kept state (budget-matched).
    if n_generated_traj > 0 and strategy in ("random_tube", "tube_select", "mined_ic", "uniform_select"):
        n_generated = n_generated_traj * steps
        states0, params = make_strategy_transitions(
            strategy, pde, surrogate, anchors, n_generated, rng, device, solver_batch_size,
            tube_rho_min, tube_rho_max, tube_kmax, candidate_factor,
        )
        next_state = safe_step_transitions(
            pde, states0, params, batch_size=solver_batch_size, progress_label=f"{strategy} transitions"
        )
        good = finite_rows(states0, params, next_state)
        n_bad = int((~good).sum())
        if n_bad > 0:
            print(f"WARNING: {strategy} produced {n_bad} nonfinite transitions; replacing with uniform.", file=sys.stderr)
            fb = make_uniform_pool(pde, max(1, int(np.ceil(n_bad / max(steps, 1)))), 1, rng)
            states0[~good] = fb.states[:n_bad]
            params[~good] = fb.params[:n_bad]
            next_state[~good] = fb.next_states[:n_bad]
        src = np.ones(n_generated, dtype=np.int8)
        if n_bad > 0:
            src[~good] = 2
        pools.append(
            TransitionPool(
                states0.astype(np.float32), params.astype(np.float32), next_state.astype(np.float32),
                source=src,
            )
        )
        return _merge_pools(pools)
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
            states_attempt, params_attempt, _ = generate_states_and_params(
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
                previous=previous,
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
        valid_bins = []
        valid_sources = []
        attempts = 0
        max_attempts = 10

        while needed > 0 and attempts < max_attempts:
            attempts += 1
            states_attempt, params_attempt, bins_attempt = generate_states_and_params(
                ddpm,
                needed,
                ddpm_mode,
                previous.losses if ddpm_mode == "conditional_loss" else None,
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
                surrogate=surrogate,
                candidate_factor=candidate_factor,
                tv_gate_threshold=tv_gate_threshold,
                anchors_phys=anchors,
                sample_mode=sample_mode,
                edit_t0=edit_t0,
                previous=previous,
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
                valid_bins.append(bins_attempt[good_indices])
                valid_sources.append(np.ones(len(good_indices), dtype=np.int8))
                needed -= len(good_indices)

        if needed > 0:
            print(
                f"WARNING: DDPM generation or transition simulation produced NaNs/Infs "
                f"after {max_attempts} attempts. Falling back to uniform sampling for "
                f"{needed} transitions.",
                file=sys.stderr,
            )
            fallback_pool = make_uniform_pool(pde, needed, 1, rng)
            valid_states0.append(fallback_pool.states)
            valid_params.append(fallback_pool.params)
            valid_next_states.append(fallback_pool.next_states)
            valid_bins.append(np.full(len(fallback_pool), -1, dtype=np.int64))
            valid_sources.append(np.full(len(fallback_pool), 2, dtype=np.int8))

        states0 = np.concatenate(valid_states0, axis=0)[:n_generated].astype(np.float32)
        params = np.concatenate(valid_params, axis=0)[:n_generated].astype(np.float32)
        next_state = np.concatenate(valid_next_states, axis=0)[:n_generated].astype(np.float32)
        bins = np.concatenate(valid_bins, axis=0)[:n_generated].astype(np.int64)
        sources = np.concatenate(valid_sources, axis=0)[:n_generated].astype(np.int8)

        pools.append(
            TransitionPool(
                states0,
                params,
                next_state,
                source=sources,
                target_bins=bins,
            )
        )
    elif n_generated_traj > 0:
        raise ValueError(f"Unknown ddpm.generated_pool_mode {generated_pool_mode!r}")
    return _merge_pools(pools)


def _merge_pools(pools: list) -> TransitionPool:
    states = np.concatenate([p.states for p in pools], axis=0)
    params = np.concatenate([p.params for p in pools], axis=0)
    next_states = np.concatenate([p.next_states for p in pools], axis=0)
    source = np.concatenate(
        [p.source if p.source is not None else np.zeros(len(p), dtype=np.int8) for p in pools], axis=0
    )
    proposed = np.full((len(states),), np.nan, dtype=np.float32)
    bins = np.full((len(states),), -1, dtype=np.int64)
    offset = 0
    for p in pools:
        if p.proposed_losses is not None:
            proposed[offset : offset + len(p)] = p.proposed_losses.reshape(-1)
        if p.target_bins is not None:
            bins[offset : offset + len(p)] = np.asarray(p.target_bins).reshape(-1)
        offset += len(p)
    return TransitionPool(
        states, params, next_states, source=source, proposed_losses=proposed, target_bins=bins
    )
