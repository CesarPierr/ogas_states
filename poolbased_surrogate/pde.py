from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .config import PDEConfig


@dataclass
class PDEParams:
    value: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.value], dtype=np.float32)


class PDE1D:
    def __init__(self, cfg: PDEConfig):
        self.cfg = cfg
        self.name = cfg.name.lower()
        self.n = int(cfg.resolution)
        self.dt = float(cfg.dt)
        self.x = np.linspace(0.0, 1.0, self.n, endpoint=False, dtype=np.float32)
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=1.0 / self.n)

    @property
    def param_range(self) -> tuple[float, float]:
        if self.name == "burgers":
            return self.cfg.viscosity_range
        if self.name == "advection":
            return self.cfg.velocity_range
        if self.name == "diffusion":
            return self.cfg.diffusion_range
        if self.name in {"ks", "kuramoto_sivashinsky"}:
            return (0.0, 1.0)
        raise ValueError(f"Unknown PDE {self.name}")

    def sample_params_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        low, high = self.param_range
        return rng.uniform(low, high, size=(n, 1)).astype(np.float32)

    def sample_params_halton(self, n: int, seed: int) -> np.ndarray:
        low, high = self.param_range
        sampler = qmc.Halton(d=1, scramble=True, seed=seed)
        u = sampler.random(n)
        return (low + (high - low) * u).astype(np.float32)

    def sample_ic_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        modes = int(self.cfg.ic.get("modes", 5))
        amp = float(self.cfg.ic.get("amplitude", 1.0))
        states = []
        for _ in range(n):
            u = np.zeros_like(self.x, dtype=np.float32)
            for m in range(1, modes + 1):
                a = rng.normal(0.0, amp / m)
                b = rng.normal(0.0, amp / m)
                u += a * np.sin(2 * np.pi * m * self.x) + b * np.cos(2 * np.pi * m * self.x)
            u = u / (np.max(np.abs(u)) + 1e-6)
            states.append(u[None, :])
        return np.stack(states).astype(np.float32)

    def sample_ic_halton(self, n: int, seed: int) -> np.ndarray:
        modes = int(self.cfg.ic.get("modes", 5))
        amp = float(self.cfg.ic.get("amplitude", 1.0))
        sampler = qmc.Halton(d=2 * modes, scramble=True, seed=seed)
        raw = 2.0 * sampler.random(n) - 1.0
        states = []
        for row in raw:
            u = np.zeros_like(self.x, dtype=np.float32)
            for m in range(1, modes + 1):
                a = amp * row[2 * (m - 1)] / m
                b = amp * row[2 * (m - 1) + 1] / m
                u += a * np.sin(2 * np.pi * m * self.x) + b * np.cos(2 * np.pi * m * self.x)
            u = u / (np.max(np.abs(u)) + 1e-6)
            states.append(u[None, :])
        return np.stack(states).astype(np.float32)

    def simulate(self, states: np.ndarray, params: np.ndarray, steps: int) -> np.ndarray:
        traj = np.empty((states.shape[0], steps + 1, 1, self.n), dtype=np.float32)
        traj[:, 0] = states
        u = states[:, 0].astype(np.float64)
        for t in range(steps):
            u = self.step(u, params[:, 0])
            traj[:, t + 1, 0] = u.astype(np.float32)
        return traj

    def step(self, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        if self.name == "advection":
            return self._advection(u, p)
        if self.name == "diffusion":
            return self._diffusion(u, p)
        if self.name == "burgers":
            return self._burgers(u, p)
        if self.name in {"ks", "kuramoto_sivashinsky"}:
            return self._ks(u)
        raise ValueError(f"Unknown PDE {self.name}")

    def _deriv(self, u: np.ndarray, order: int) -> np.ndarray:
        u_hat = np.fft.fft(u, axis=-1)
        return np.fft.ifft((1j * self.k) ** order * u_hat, axis=-1).real

    def _advection(self, u: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        return u - self.dt * velocity[:, None] * self._deriv(u, 1)

    def _diffusion(self, u: np.ndarray, diffusivity: np.ndarray) -> np.ndarray:
        return u + self.dt * diffusivity[:, None] * self._deriv(u, 2)

    def _burgers(self, u: np.ndarray, viscosity: np.ndarray) -> np.ndarray:
        ux = self._deriv(u, 1)
        uxx = self._deriv(u, 2)
        out = u + self.dt * (-u * ux + viscosity[:, None] * uxx)
        return np.clip(out, -5.0, 5.0)

    def _ks(self, u: np.ndarray) -> np.ndarray:
        ux = self._deriv(u, 1)
        uxx = self._deriv(u, 2)
        uxxxx = self._deriv(u, 4)
        out = u + self.dt * (-u * ux - uxx - uxxxx)
        return np.clip(out, -10.0, 10.0)
