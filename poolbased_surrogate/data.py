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

    @classmethod
    def from_trajectories(cls, trajectories: np.ndarray, params: np.ndarray) -> "TransitionPool":
        n_traj, steps_plus_one = trajectories.shape[:2]
        steps = steps_plus_one - 1
        states = trajectories[:, :-1].reshape(n_traj * steps, *trajectories.shape[2:])
        next_states = trajectories[:, 1:].reshape(n_traj * steps, *trajectories.shape[2:])
        repeated_params = np.repeat(params, steps, axis=0)
        return cls(states.astype(np.float32), repeated_params.astype(np.float32), next_states.astype(np.float32))

    def replace(self, other: "TransitionPool") -> None:
        self.states = other.states
        self.params = other.params
        self.next_states = other.next_states
        self.losses = other.losses

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


class StateLossDataset(Dataset):
    def __init__(self, states: np.ndarray, losses: np.ndarray):
        self.states = torch.from_numpy(states.astype(np.float32))
        losses = np.asarray(losses, dtype=np.float32).reshape(-1, 1)
        self.losses = torch.from_numpy(losses)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int):
        return self.states[idx], self.losses[idx]
