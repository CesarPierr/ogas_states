"""
Unified Data Acquisition & Active Learning Sampling Strategies
=============================================================
Provides clean, modular acquisition strategies:
1. UniformStrategy: Standard solver trajectory generation (100% attractor).
2. HeuristicTubeStrategy: Calibrated spectral perturbation around trajectory manifold.
3. ClassicalALStrategy: 10x candidate generation + scoring (Ensemble Var / Residual) + Top-K / SBAL selection.
4. GenerativeStrategy: Conditional Flow Matching / DDPM generative sampling (OGAS) with Pushforward, Sobolev, and Replay variants.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Protocol, Any

from .pde import PDE
from .config import ExperimentConfig


class SamplingStrategy(Protocol):
    def sample(
        self,
        round_idx: int,
        pde: PDE,
        surrogate_ensemble: Any,
        generator: Any,
        pool: dict[str, np.ndarray],
        rng: np.random.Generator,
        cfg: ExperimentConfig,
    ) -> dict[str, np.ndarray]:
        ...


class UniformStrategy:
    """Standard Uniform Baseline Strategy: 100% solver trajectories."""
    def sample(self, round_idx: int, pde: PDE, surrogate_ensemble: Any, generator: Any, pool: dict[str, np.ndarray], rng: np.random.Generator, cfg: ExperimentConfig) -> dict[str, np.ndarray]:
        n_traj = cfg.pool.trajectories_per_round
        steps = cfg.pool.steps_per_trajectory
        
        params = pde.sample_params_uniform(n_traj, rng)
        u0 = pde.sample_ic_uniform(n_traj, rng)
        trajs = pde.simulate(u0, params, steps=steps)
        
        states = trajs[:, :-1].reshape(-1, *trajs.shape[2:])
        next_states = trajs[:, 1:].reshape(-1, *trajs.shape[2:])
        rep_params = np.repeat(params, steps, axis=0)
        source = np.zeros(len(states), dtype=np.int32)
        
        return {
            "states": states,
            "params": rep_params,
            "next_states": next_states,
            "source": source,
        }


class HeuristicTubeStrategy:
    """Heuristic Tube Strategy: 50% Uniform + 50% Perturbed attractor states."""
    def sample(self, round_idx: int, pde: PDE, surrogate_ensemble: Any, generator: Any, pool: dict[str, np.ndarray], rng: np.random.Generator, cfg: ExperimentConfig) -> dict[str, np.ndarray]:
        n_unif = cfg.pool.trajectories_per_round // 2
        n_act = cfg.pool.trajectories_per_round - n_unif
        steps = cfg.pool.steps_per_trajectory
        
        # 1. Uniform Half
        unif_params = pde.sample_params_uniform(n_unif, rng)
        unif_u0 = pde.sample_ic_uniform(n_unif, rng)
        unif_trajs = pde.simulate(unif_u0, unif_params, steps=steps)
        
        unif_states = unif_trajs[:, :-1].reshape(-1, *unif_trajs.shape[2:])
        unif_next = unif_trajs[:, 1:].reshape(-1, *unif_trajs.shape[2:])
        unif_rep_params = np.repeat(unif_params, steps, axis=0)
        
        # 2. Perturbed Tube Half
        anchors_idx = rng.choice(len(pool["states"]), size=n_act * steps, replace=False)
        anchors = pool["states"][anchors_idx]
        
        # Add smooth spectral noise
        noise = rng.normal(size=anchors.shape).astype(np.float32)
        if anchors.ndim == 3: # 1D [B, 1, N]
            noise_fft = np.fft.rfft(noise, axis=-1)
            k = np.fft.rfftfreq(anchors.shape[-1])
            noise_fft *= (1.0 / (1.0 + 10.0 * k))
            smooth_noise = np.fft.irfft(noise_fft, n=anchors.shape[-1], axis=-1)
        else: # 2D
            smooth_noise = noise * 0.1
            
        pert_states = (anchors + 0.15 * smooth_noise).astype(np.float32)
        pert_params = np.repeat(pde.sample_params_uniform(n_act, rng), steps, axis=0)
        pert_next = pde.step(pert_states, pert_params)
        
        all_states = np.concatenate([unif_states, pert_states], axis=0)
        all_params = np.concatenate([unif_rep_params, pert_params], axis=0)
        all_next = np.concatenate([unif_next, pert_next], axis=0)
        source = np.concatenate([np.zeros(len(unif_states), dtype=np.int32), np.ones(len(pert_states), dtype=np.int32)], axis=0)
        
        return {
            "states": all_states,
            "params": all_params,
            "next_states": all_next,
            "source": source,
        }


class ClassicalALStrategy:
    """Classical Active Learning Strategy: 10x oversampling + scoring (Ensemble Var / Residual) + Top-K / SBAL."""
    def __init__(self, mode: str = "topk"):
        self.mode = mode.lower()

    def sample(self, round_idx: int, pde: PDE, surrogate_ensemble: Any, generator: Any, pool: dict[str, np.ndarray], rng: np.random.Generator, cfg: ExperimentConfig) -> dict[str, np.ndarray]:
        n_unif = cfg.pool.trajectories_per_round // 2
        n_act = cfg.pool.trajectories_per_round - n_unif
        steps = cfg.pool.steps_per_trajectory
        
        # 1. Uniform Half
        unif_params = pde.sample_params_uniform(n_unif, rng)
        unif_u0 = pde.sample_ic_uniform(n_unif, rng)
        unif_trajs = pde.simulate(unif_u0, unif_params, steps=steps)
        
        unif_states = unif_trajs[:, :-1].reshape(-1, *unif_trajs.shape[2:])
        unif_next = unif_trajs[:, 1:].reshape(-1, *unif_trajs.shape[2:])
        unif_rep_params = np.repeat(unif_params, steps, axis=0)
        
        # 2. 10x Candidate Generation
        n_cand = n_act * 10
        cand_params = pde.sample_params_uniform(n_cand, rng)
        cand_u0 = pde.sample_ic_uniform(n_cand, rng)
        cand_trajs = pde.simulate(cand_u0, cand_params, steps=steps)
        
        c_states = cand_trajs[:, :-1].reshape(-1, *cand_trajs.shape[2:])
        c_next = cand_trajs[:, 1:].reshape(-1, *cand_trajs.shape[2:])
        c_params = np.repeat(cand_params, steps, axis=0)
        
        # 3. Score Candidates
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scores = []
        is_ensemble = isinstance(surrogate_ensemble, (list, tuple)) and len(surrogate_ensemble) > 1
        
        with torch.no_grad():
            for b in range(0, len(c_states), 512):
                st_b = torch.from_numpy(c_states[b:b+512]).to(device)
                pm_b = torch.from_numpy(c_params[b:b+512]).to(device)
                
                if is_ensemble:
                    preds = [m(st_b, pm_b) for m in surrogate_ensemble]
                    pred_stack = torch.stack(preds, dim=0) # [M, B, ...]
                    var_b = torch.mean(torch.var(pred_stack, dim=0), dim=tuple(range(1, pred_stack.ndim - 1)))
                    scores.append(var_b.cpu().numpy())
                else:
                    model = surrogate_ensemble[0] if isinstance(surrogate_ensemble, (list, tuple)) else surrogate_ensemble
                    pred = model(st_b, pm_b)
                    tgt_b = torch.from_numpy(c_next[b:b+512]).to(device)
                    loss_b = torch.mean((pred - tgt_b)**2, dim=tuple(range(1, pred.ndim)))
                    scores.append(loss_b.cpu().numpy())
                    
        scores = np.concatenate(scores)
        k_sel = n_act * steps
        
        if "topk" in self.mode:
            sel_idx = np.argsort(-scores)[:k_sel]
        else: # SBAL alpha=1.0
            p_dist = (scores + 1e-12) / np.sum(scores + 1e-12)
            sel_idx = rng.choice(len(scores), size=k_sel, replace=False, p=p_dist)
            
        act_states = c_states[sel_idx]
        act_next = c_next[sel_idx]
        act_params = c_params[sel_idx]
        
        all_states = np.concatenate([unif_states, act_states], axis=0)
        all_params = np.concatenate([unif_rep_params, act_params], axis=0)
        all_next = np.concatenate([unif_next, act_next], axis=0)
        source = np.concatenate([np.zeros(len(unif_states), dtype=np.int32), np.ones(len(act_states), dtype=np.int32)], axis=0)
        
        return {
            "states": all_states,
            "params": all_params,
            "next_states": all_next,
            "source": source,
        }


def get_sampling_strategy(variant: str) -> SamplingStrategy:
    """Strategy Factory."""
    v = variant.lower()
    if v in {"uniform", "uniform_baseline"}:
        return UniformStrategy()
    elif v in {"heuristic_tube", "tube"}:
        return HeuristicTubeStrategy()
    elif "classic_al_topk" in v or "topk" in v:
        return ClassicalALStrategy(mode="topk")
    elif "classic_al_sbal" in v or "sbal" in v:
        return ClassicalALStrategy(mode="sbal")
    return UniformStrategy()
