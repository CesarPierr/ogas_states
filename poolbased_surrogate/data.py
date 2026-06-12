from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TransitionPool:
    states: np.ndarray
    params: np.ndarray
    next_states: np.ndarray
    losses: np.ndarray | None = None
    # source codes: 0 = uniform trajectory, 1 = generated/strategy state, 2 = fallback-to-uniform
    source: np.ndarray | None = None
    proposed_losses: np.ndarray | None = None
    pretrain_losses: np.ndarray | None = None
    uncertainty: np.ndarray | None = None  # per-transition ensemble disagreement (post-train)
    pretrain_uncertainty: np.ndarray | None = None  # disagreement before training on this pool
    target_bins: np.ndarray | None = None  # quantile bin actually requested per generated state (-1 = n/a)

    @classmethod
    def from_trajectories(
        cls,
        trajectories: np.ndarray,
        params: np.ndarray,
        source: int = 0,
    ) -> "TransitionPool":
        n_traj, steps_plus_one = trajectories.shape[:2]
        steps = steps_plus_one - 1
        states = trajectories[:, :-1].reshape(n_traj * steps, *trajectories.shape[2:])
        next_states = trajectories[:, 1:].reshape(n_traj * steps, *trajectories.shape[2:])
        repeated_params = np.repeat(params, steps, axis=0)
        source_values = np.full((n_traj * steps,), source, dtype=np.int8)
        return cls(
            states.astype(np.float32),
            repeated_params.astype(np.float32),
            next_states.astype(np.float32),
            source=source_values,
        )

    def replace(self, other: "TransitionPool") -> None:
        self.states = other.states
        self.params = other.params
        self.next_states = other.next_states
        self.losses = other.losses
        self.source = other.source
        self.proposed_losses = other.proposed_losses
        self.pretrain_losses = other.pretrain_losses
        self.uncertainty = other.uncertainty
        self.pretrain_uncertainty = other.pretrain_uncertainty
        self.target_bins = other.target_bins

    def __len__(self) -> int:
        return int(self.states.shape[0])


class TransitionDataset(Dataset):
    def __init__(self, pool: TransitionPool):
        self.pool = pool

    def __len__(self) -> int:
        return len(self.pool)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.pool.states[idx]),
            torch.from_numpy(self.pool.params[idx]),
            torch.from_numpy(self.pool.next_states[idx]),
        )
