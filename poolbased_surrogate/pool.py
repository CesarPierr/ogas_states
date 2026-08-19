"""
Active Learning Transition Pool Management
==========================================
Delegates pool creation to the modular Strategy registry in strategies.py.
Preserves backwards compatibility for make_uniform_pool and make_mixed_pool.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from .config import ExperimentConfig
from .pde import PDE
from .strategies import (
    UniformStrategy,
    HeuristicTubeStrategy,
    ClassicalALStrategy,
    get_sampling_strategy,
)


def make_uniform_pool(
    pde: PDE,
    n_trajectories: int,
    steps_per_trajectory: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate initial Round 0 pool using standard solver trajectories."""
    params = pde.sample_params_uniform(n_trajectories, rng)
    u0 = pde.sample_ic_uniform(n_trajectories, rng)
    trajs = pde.simulate(u0, params, steps=steps_per_trajectory)
    
    states = trajs[:, :-1].reshape(-1, *trajs.shape[2:])
    next_states = trajs[:, 1:].reshape(-1, *trajs.shape[2:])
    rep_params = np.repeat(params, steps_per_trajectory, axis=0)
    source = np.zeros(len(states), dtype=np.int32)
    
    return {
        "states": states.astype(np.float32),
        "params": rep_params.astype(np.float32),
        "next_states": next_states.astype(np.float32),
        "source": source,
    }


def make_mixed_pool(
    round_idx: int,
    pde: PDE,
    surrogate: Any,
    generator: Any,
    current_pool: dict[str, np.ndarray],
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    prev_losses: np.ndarray | None = None,
    prev_uncertainties: np.ndarray | None = None,
    pool_states: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sample next round transitions according to configured strategy."""
    strategy = get_sampling_strategy(cfg.pool.variant)
    new_data = strategy.sample(
        round_idx=round_idx,
        pde=pde,
        surrogate_ensemble=surrogate,
        generator=generator,
        pool=current_pool,
        rng=rng,
        cfg=cfg,
    )
    
    # Concatenate with existing pool
    merged = {}
    for k in ["states", "params", "next_states", "source"]:
        merged[k] = np.concatenate([current_pool[k], new_data[k]], axis=0)
    return merged
