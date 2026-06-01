from __future__ import annotations

import numpy as np

from .al4pde_bridge import AL4PDE1D
from .config import PDEConfig


class PDE1D:
    """Thin wrapper over the AL4PDE backend, the only supported solver/sampler path."""

    def __init__(self, cfg: PDEConfig):
        if cfg.backend.lower() != "al4pde":
            raise ValueError(
                f"Only the al4pde backend is supported, got backend={cfg.backend!r}"
            )
        self.cfg = cfg
        self.name = cfg.name.lower()
        self.resolution = int(cfg.resolution)
        self.al4pde = AL4PDE1D(cfg)

    @property
    def param_ranges(self) -> list[tuple[float, float]]:
        return self.al4pde.param_ranges

    @property
    def param_log_scale(self) -> list[bool]:
        return self.al4pde.param_log_scale

    def sample_params_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.al4pde.sample_params(n, int(rng.integers(0, 2**31 - 1)))

    def sample_params_halton(self, n: int, seed: int) -> np.ndarray:
        return self.al4pde.sample_params(n, seed)

    def sample_ic_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.al4pde.sample_ic(n, int(rng.integers(0, 2**31 - 1)))

    def sample_ic_halton(self, n: int, seed: int) -> np.ndarray:
        return self.al4pde.sample_ic(n, seed)

    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        return self.al4pde.normalize_params(params)

    def denormalize_params(self, normed: np.ndarray) -> np.ndarray:
        return self.al4pde.denormalize_params(normed)

    def simulate(self, states: np.ndarray, params: np.ndarray, steps: int, apply_warmup: bool = True) -> np.ndarray:
        return self.al4pde.simulate(states, params, steps, apply_warmup=apply_warmup)

    def step(self, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Advance states by one outer step.  Never applies warmup — used for
        DDPM-generated states where the IC may not be on the attractor."""
        states = np.asarray(u, dtype=np.float32)[:, None, :]
        return self.al4pde.simulate(states, p.astype(np.float32), 1, apply_warmup=False)[:, 1, 0]
