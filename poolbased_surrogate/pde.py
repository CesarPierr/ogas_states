"""
Unified PDE Interface: AL4PDE 1D & 2D Backends
=============================================
Provides a single, clean PDE class supporting:
- 1D PDEs: Burgers, Kuramoto-Sivashinsky (KS)
- 2D PDEs: Navier-Stokes Kolmogorov Flow (CFDSim)

Follows standard NumPy / PyTorch tensor contracts across all dimensions.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Sequence
import numpy as np
import torch

from .config import PDEConfig

DEFAULT_EXTERNAL = Path("/bettik/PROJECTS/pr-melissa/cesarpi-ext/external")
LEONARDO_EXTERNAL = Path("/leonardo_scratch/fast/EUHPC_D36_033/ogas_external")


def ensure_al4pde_paths() -> None:
    """Ensure AL4PDE, PDEArena, and JAX-CFD are in sys.path."""
    candidate_bases = [
        Path(os.environ.get("OGAS_EXTERNAL", "")),
        LEONARDO_EXTERNAL,
        DEFAULT_EXTERNAL,
    ]
    submodules = ["al4pde", "pdearena", "jax-cfd"]
    for base in candidate_bases:
        if base and base.exists():
            for sub in submodules:
                sub_p = base / sub
                if sub_p.exists() and str(sub_p) not in sys.path:
                    sys.path.insert(0, str(sub_p))

    roots = [
        Path(os.environ.get("AL4PDE_ROOT", DEFAULT_EXTERNAL / "al4pde")),
        Path(os.environ.get("PDEARENA_ROOT", DEFAULT_EXTERNAL / "pdearena")),
        Path(os.environ.get("JAX_CFD_ROOT", DEFAULT_EXTERNAL / "jax-cfd")),
    ]
    for root in roots:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


def torch_generator(seed: int) -> torch.Generator:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(int(seed))


class SqueezeTensorWrapper:
    def __init__(self, tensor: torch.Tensor, is_params: bool = False):
        self.tensor = tensor
        self.is_params = is_params

    def squeeze(self, *args, **kwargs):
        if self.is_params and self.tensor.ndim == 2:
            return self.tensor
        return self.tensor.squeeze(*args, **kwargs)

    def cpu(self):
        return self.tensor.cpu()

    def numpy(self):
        return self.tensor.cpu().numpy()


class PDE:
    """Unified PDE Simulator & Sampler for 1D and 2D equations."""

    def __init__(self, cfg: PDEConfig):
        ensure_al4pde_paths()
        self.cfg = cfg
        self.name = cfg.name.lower()
        self.resolution = int(cfg.resolution)
        self.dt = float(cfg.dt)
        self.length = float(cfg.domain_length)
        self.n_substeps = max(1, int(getattr(cfg, "n_substeps", 1)))
        self.n_warmup = max(0, int(getattr(cfg, "n_warmup", 0)))
        self.spatial_dim = 2 if self.name in {"ns2d", "cfd2d", "2d_ns_rand", "kolmogorov2d"} else 1
        self._sim_cache: dict[int, object] = {}

    @property
    def is_ks(self) -> bool:
        return self.name in {"ks", "kuramoto_sivashinsky"}

    @property
    def is_burgers(self) -> bool:
        return self.name == "burgers"

    @property
    def is_2d(self) -> bool:
        return self.spatial_dim == 2

    @property
    def param_ranges(self) -> list[tuple[float, float]]:
        if self.cfg.param_ranges is not None:
            return [tuple(float(v) for v in pair) for pair in self.cfg.param_ranges]
        if self.is_burgers:
            return [tuple(float(v) for v in self.cfg.viscosity_range)]
        if self.is_ks:
            return [(0.5, 4.0), (0.1, 100.0)]
        if self.is_2d:
            return [(1e-4, 0.1), (1e-4, 0.1)]
        raise ValueError(f"Unknown PDE: {self.name!r}")

    @property
    def param_log_scale(self) -> list[bool]:
        if self.cfg.param_log_scale is not None:
            return list(self.cfg.param_log_scale)
        if self.is_2d:
            return [True, True]
        return [self.is_burgers] + [False] * (len(self.param_ranges) - 1)

    def _param_generator(self):
        from al4pde.tasks.param_gen import PDEParamGenerator
        lows = [r[0] for r in self.param_ranges]
        highs = [r[1] for r in self.param_ranges]
        return PDEParamGenerator(lows, highs, self.param_log_scale)

    def sample_params(self, n: int, seed: int) -> np.ndarray:
        gen = self._param_generator()
        gen.set_rng(torch_generator(seed))
        with torch.no_grad():
            params = gen.get_pde_params(gen.get_normed_pde_params(n))
        return params.detach().cpu().numpy().astype(np.float32)

    def sample_params_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sample_params(n, int(rng.integers(0, 2**31 - 1)))

    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Map physical PDE parameters to normalized [0, 1] space."""
        params = np.asarray(params, dtype=np.float64)
        out = np.empty_like(params)
        for j, ((low, high), log_scale) in enumerate(zip(self.param_ranges, self.param_log_scale)):
            col = params[:, j]
            if high <= low:
                out[:, j] = 0.0
            elif log_scale:
                out[:, j] = np.log(np.clip(col, low, high) / low) / np.log(high / low)
            else:
                out[:, j] = (col - low) / (high - low)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def denormalize_params(self, normed: np.ndarray) -> np.ndarray:
        """Map normalized [0, 1] parameters back to physical space."""
        normed = np.clip(np.asarray(normed, dtype=np.float64), 0.0, 1.0)
        out = np.empty_like(normed)
        for j, ((low, high), log_scale) in enumerate(zip(self.param_ranges, self.param_log_scale)):
            u = normed[:, j]
            if high <= low:
                out[:, j] = low
            elif log_scale:
                out[:, j] = low * (high / low) ** u
            else:
                out[:, j] = low + u * (high - low)
        return out.astype(np.float32)

    def _ic_generator(self):
        if self.is_ks:
            from al4pde.tasks.ic_gen.ic_gen_ks import ICGenKS
            return ICGenKS(
                N=int(self.cfg.ic.get("N", 10)),
                nx=self.resolution,
                lmin=int(self.cfg.ic.get("lmin", 1)),
                lmax=int(self.cfg.ic.get("lmax", 10)),
                amp_max=float(self.cfg.ic.get("amp_max", self.cfg.ic.get("amplitude", 1.0))),
                requires_grad=False,
                length=float(self.cfg.ic.get("length", 1.0)),
                in_fourier_domain=bool(self.cfg.ic.get("in_fourier_domain", False)),
            )
        if self.is_burgers:
            from al4pde.tasks.ic_gen.ic_gen_burgers import ICGenBurgers
            return ICGenBurgers(
                k_tot=int(self.cfg.ic.get("k_tot", 4)),
                num_choice_k=int(self.cfg.ic.get("num_choice_k", 2)),
                xL=0.0,
                xR=self.length,
                nx=self.resolution,
                requires_grad=False,
            )
        if self.is_2d:
            from al4pde.tasks.ic_gen.ic_gen_2d_ns_rand import ICGenNSRand
            return ICGenNSRand(
                k_tot=4,
                xL=0.0,
                xR=self.length,
                yL=0.0,
                yR=self.length,
                zL=0.0,
                zR=1.0,
                nx=self.resolution,
                ny=self.resolution,
                nz=1,
                gamma=1.6666666666666667,
                mach_min=0.1,
                mach_max=1.0,
                d0Min=1e-1,
                d0Max=1e1,
                T0Min=1e-1,
                T0Max=1e1,
                delDMin=1.3e-2,
                delDMax=0.26,
                delPMin=4e-2,
                delPMax=0.8,
                init_field_type="rand",
                constrain_max=True,
                requires_grad=False,
            )
        raise ValueError(f"Unknown PDE: {self.name!r}")

    def sample_ic(self, n: int, seed: int) -> np.ndarray:
        gen = self._ic_generator()
        gen.set_rng(torch_generator(seed))
        params = torch.zeros((n, len(self.param_ranges)), dtype=torch.float32)
        with torch.no_grad():
            ic_params = gen.initialize_ic_params(n)
            states = gen.generate_initial_conditions(ic_params, params)
        if self.is_2d:
            # Velocity components [vx, vy] -> shape [B, 2, H, W]
            vel = states[..., [1, 2]].detach().cpu().numpy()
            return vel[:, :, :, 0].transpose(0, 3, 1, 2).astype(np.float32)
        # 1D IC shape [B, 1, N]
        return states.detach().cpu().numpy().transpose(0, 3, 1, 2)[:, :, :, 0].astype(np.float32)

    def sample_ic_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sample_ic(n, int(rng.integers(0, 2**31 - 1)))

    def grid(self, n: int = 1) -> torch.Tensor:
        grid = self._ic_generator().get_grid(n)
        return grid.detach().cpu().float()

    def simulator(self, steps: int):
        n_inner = steps * self.n_substeps
        fin_time = self.dt * n_inner
        if self.is_ks:
            from al4pde.tasks.sim.ks_jax import ParametricKSJaxSim
            return ParametricKSJaxSim(
                ini_time=0.0,
                dt=self.dt,
                fin_time=fin_time,
                pde_name="KSVarLVIC",
                L=self.length,
            )
        if self.is_burgers:
            from al4pde.tasks.sim.burgers import BurgersSim
            return BurgersSim(
                ini_time=0.0,
                fin_time=fin_time,
                dt=self.dt,
                CFL=float(self.cfg.ic.get("CFL", 2.5e-1)),
                show_steps=int(self.cfg.ic.get("show_steps", 100)),
                if_norm=False,
                if_second_order=float(self.cfg.ic.get("if_second_order", 1.0)),
            )
        if self.is_2d:
            from al4pde.tasks.sim.cfd import CFDSim
            return CFDSim(
                ini_time=0.0,
                fin_time=fin_time,
                dt=self.dt,
                CFL=0.3,
                show_steps=100,
                if_second_order=1.0,
                bc="periodic",
                gamma=1.6666666666666667,
                p_floor=1e-4,
                spatial_dim=2,
                same_eta_zeta=False,
            )
        raise ValueError(f"Unknown PDE: {self.name!r}")

    def simulate(self, states: np.ndarray, params: np.ndarray, steps: int, apply_warmup: bool = True) -> np.ndarray:
        """Simulate trajectories. Returns [B, steps+1, C, N] (1D) or [B, steps+1, 2, H, W] (2D)."""
        B = len(states)
        if B == 0:
            return np.empty((0, steps + 1, 2 if self.is_2d else 1, self.resolution), dtype=np.float32)

        sim = self.simulator(steps)
        
        if self.is_2d:
            full_states = np.zeros((B, self.resolution, self.resolution, 1, 4), dtype=np.float32)
            full_states[..., 0] = 1.0 # density
            full_states[..., 1] = states[:, 0, :, :, None] # vx
            full_states[..., 2] = states[:, 1, :, :, None] # vy
            full_states[..., 3] = 1.0 # pressure
            
            gen = self._ic_generator()
            grid_2d = gen.get_grid(B)
            dx = 1.0 / self.resolution
            grid_3d = torch.zeros((B, self.resolution, self.resolution, 1, 3), device=grid_2d.device)
            grid_3d[..., 0] = grid_2d[..., 0:1]
            grid_3d[..., 1] = grid_2d[..., 1:2]
            grid_3d[..., 2] = dx
            ic_t = torch.from_numpy(full_states)
            pm_t = torch.from_numpy(params.astype(np.float32))
            
            with torch.no_grad():
                res = sim(ic_t, pm_t, grid_3d)
                traj = res[0]
            traj_np = traj.detach().cpu().numpy()
            if traj_np.ndim == 5 and traj_np.shape[1] == steps + 1:
                return traj_np[..., [1, 2]].transpose(0, 1, 4, 2, 3).astype(np.float32)
            elif traj_np.ndim == 5 and traj_np.shape[3] == steps + 1:
                return traj_np[..., [1, 2]].transpose(0, 3, 4, 1, 2).astype(np.float32)
            return traj_np[..., [1, 2]].transpose(0, 1, 4, 2, 3).astype(np.float32)

        # 1D Simulation
        if self.is_burgers:
            trajs = []
            for i in range(B):
                ic_i = torch.from_numpy(states[i:i+1, 0, :, None, None].astype(np.float32))
                pm_i = torch.from_numpy(params[i:i+1].astype(np.float32))
                grid_i = self.grid(1)
                with torch.no_grad():
                    traj_i, _, _ = sim(SqueezeTensorWrapper(ic_i), SqueezeTensorWrapper(pm_i, is_params=True), grid_i)
                trajs.append(traj_i.detach().cpu().numpy().transpose(0, 1, 3, 2).astype(np.float32))
            full = np.concatenate(trajs, axis=0)
        else:
            ic = torch.from_numpy(states[:, 0, :, None, None].astype(np.float32))
            pm = torch.from_numpy(params.astype(np.float32))
            grid = self.grid(B)
            with torch.no_grad():
                traj, _, _ = sim(ic, pm, grid)
            full = traj.detach().cpu().numpy().transpose(0, 1, 3, 2).astype(np.float32)

        if self.n_substeps > 1:
            return full[:, ::self.n_substeps]
        return full

    def step(self, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Advance single transitions by one time step."""
        if self.is_2d:
            return self.simulate(u, p, steps=1)[:, 1]
        states = np.asarray(u, dtype=np.float32)[:, None, :]
        return self.simulate(states, p.astype(np.float32), 1, apply_warmup=False)[:, 1, 0]


# Backwards compatibility aliases
PDE1D = PDE
PDE2D = PDE
