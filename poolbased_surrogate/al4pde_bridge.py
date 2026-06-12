from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


DEFAULT_EXTERNAL = Path("/bettik/PROJECTS/pr-melissa/cesarpi-ext/external")


def ensure_al4pde_paths() -> None:
    roots = [
        Path(os.environ.get("AL4PDE_ROOT", DEFAULT_EXTERNAL / "al4pde")),
        Path(os.environ.get("PDEARENA_ROOT", DEFAULT_EXTERNAL / "pdearena")),
        Path(os.environ.get("JAX_CFD_ROOT", DEFAULT_EXTERNAL / "jax-cfd")),
    ]
    for root in roots:
        if root.exists():
            path = str(root)
            if path not in sys.path:
                sys.path.insert(0, path)


def torch_generator(seed: int) -> torch.Generator:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(int(seed))


def numpy_to_al4pde_ic(states: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(states[:, 0, :, None, None].astype(np.float32))


def al4pde_traj_to_numpy(traj: torch.Tensor) -> np.ndarray:
    return traj.detach().cpu().numpy().transpose(0, 1, 3, 2).astype(np.float32)


class SqueezeNumpyWrapper:
    def __init__(self, arr, is_params=False):
        self.arr = arr
        self.is_params = is_params

    def squeeze(self):
        if self.is_params:
            return self.arr.reshape((1,))
        else:
            return self.arr.reshape((1, -1))


class SqueezeTensorWrapper:
    def __init__(self, tensor, is_params=False):
        self.tensor = tensor
        self.is_params = is_params

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return SqueezeNumpyWrapper(self.tensor.cpu().numpy(), self.is_params)


class AL4PDE1D:
    def __init__(self, cfg):
        ensure_al4pde_paths()
        self.cfg = cfg
        self.name = cfg.name.lower()
        self.n = int(cfg.resolution)
        self.dt = float(cfg.dt)
        self.length = float(cfg.domain_length)
        self.n_substeps = max(1, int(getattr(cfg, "n_substeps", 1)))
        self.n_warmup = max(0, int(getattr(cfg, "n_warmup", 0)))
        self._sim_cache: dict[int, object] = {}

    @property
    def is_ks(self) -> bool:
        return self.name in {"ks", "kuramoto_sivashinsky"}

    @property
    def is_burgers(self) -> bool:
        return self.name == "burgers"

    def _param_generator(self):
        from al4pde.tasks.param_gen import PDEParamGenerator

        ranges = self.param_ranges
        lows = [r[0] for r in ranges]
        highs = [r[1] for r in ranges]
        return PDEParamGenerator(lows, highs, self.param_log_scale)

    @property
    def param_ranges(self) -> list[tuple[float, float]]:
        if self.cfg.param_ranges is not None:
            return [tuple(float(v) for v in pair) for pair in self.cfg.param_ranges]
        if self.is_burgers:
            return [tuple(float(v) for v in self.cfg.viscosity_range)]
        if self.is_ks:
            return [(0.5, 4.0), (0.1, 100.0)]
        raise ValueError(f"AL4PDE backend is not configured for PDE {self.name!r}")

    @property
    def param_log_scale(self) -> list[bool]:
        if self.cfg.param_log_scale is not None:
            return list(self.cfg.param_log_scale)
        return [self.is_burgers] + [False] * (len(self.param_ranges) - 1)

    def sample_params(self, n: int, seed: int) -> np.ndarray:
        gen = self._param_generator()
        gen.set_rng(torch_generator(seed))
        with torch.no_grad():
            params = gen.get_pde_params(gen.get_normed_pde_params(n))
        return params.detach().cpu().numpy().astype(np.float32)

    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Map physical PDE parameters to AL4PDE normed space (per-dim in [0, 1])."""
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
        """Inverse of normalize_params: normed space (per-dim in [0, 1]) to physical."""
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
                nx=self.n,
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
                nx=self.n,
                requires_grad=False,
            )
        raise ValueError(f"AL4PDE backend is not configured for PDE {self.name!r}")

    def sample_ic(self, n: int, seed: int) -> np.ndarray:
        gen = self._ic_generator()
        gen.set_rng(torch_generator(seed))
        params = torch.zeros((n, len(self.param_ranges)), dtype=torch.float32)
        with torch.no_grad():
            ic_params = gen.initialize_ic_params(n)
            states = gen.generate_initial_conditions(ic_params, params)
        # AL4PDE IC format is [B, X, T=1, C=1].
        return states.detach().cpu().numpy().transpose(0, 3, 1, 2)[:, :, :, 0].astype(np.float32)

    def grid(self, n: int = 1) -> torch.Tensor:
        grid = self._ic_generator().get_grid(n)
        return grid.detach().cpu().float()

    def simulator(self, steps: int):
        # Each "outer step" (dataset transition) covers n_substeps × dt time.
        # The solver runs with the original dt so numerics are unchanged;
        # we just take n_substeps consecutive solver steps per training transition.
        n_inner = steps * self.n_substeps          # total solver steps
        fin_time = self.dt * n_inner               # total time covered
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
        raise ValueError(f"AL4PDE backend is not configured for PDE {self.name!r}")

    def _run_sim(self, states: np.ndarray, params: np.ndarray, steps: int) -> np.ndarray:
        """Run the simulator for `steps` outer steps and return shape [B, steps+1, 1, N].
        Substeps are used internally; output is subsampled to outer timesteps only."""
        if states.shape[0] == 0:
            return np.empty((0, steps + 1, 1, self.n), dtype=np.float32)
        if steps not in self._sim_cache:
            self._sim_cache[steps] = self.simulator(steps)
        sim = self._sim_cache[steps]
        if self.is_burgers:
            # Burgers solver in AL4PDE only supports batch_size=1 due to a JAX shape mismatch in adaptive dt
            trajs = []
            for i in range(len(states)):
                ic_i = numpy_to_al4pde_ic(states[i:i+1])
                pde_params_i = torch.from_numpy(params[i:i+1].astype(np.float32))
                grid_i = self.grid(1)
                wrapped_ic = SqueezeTensorWrapper(ic_i, is_params=False)
                wrapped_params = SqueezeTensorWrapper(pde_params_i, is_params=True)
                with torch.no_grad():
                    traj_i, _, _ = sim(wrapped_ic, wrapped_params, grid_i)
                trajs.append(al4pde_traj_to_numpy(traj_i))
            full = np.concatenate(trajs, axis=0)
        else:
            ic = numpy_to_al4pde_ic(states)
            pde_params = torch.from_numpy(params.astype(np.float32))
            grid = self.grid(len(states))
            with torch.no_grad():
                traj, _, _ = sim(ic, pde_params, grid)
            full = al4pde_traj_to_numpy(traj)  # [B, steps*n_sub+1, 1, N]
        if self.n_substeps > 1:
            return full[:, :: self.n_substeps]  # [B, steps+1, 1, N]
        return full

    def simulate(self, states: np.ndarray, params: np.ndarray, steps: int, apply_warmup: bool = True) -> np.ndarray:
        """Simulate `steps` outer steps.  When apply_warmup=True and n_warmup>0 the IC
        is first advanced by n_warmup outer steps before the recorded trajectory starts."""
        if states.shape[0] == 0:
            return np.empty((0, steps + 1, 1, self.n), dtype=np.float32)
        if apply_warmup and self.n_warmup > 0:
            warmed = self._run_sim(states, params, self.n_warmup)[:, -1]  # [B, 1, N]
            states = warmed
        return self._run_sim(states, params, steps)


class ExactAL4PDEUnet1D(torch.nn.Module):
    def __init__(
        self,
        param_dim: int,
        hidden_channels: int = 16,
        difference_weight: float = 0.3,
    ):
        super().__init__()
        ensure_al4pde_paths()
        from al4pde.modules.unet_cond_1d import Unet1D

        self.difference_weight = float(difference_weight)
        self.model = Unet1D(
            padding_mode="circular",
            n_input_scalar_components=1,
            n_input_vector_components=0,
            n_output_scalar_components=1,
            n_output_vector_components=0,
            time_history=1,
            time_future=1,
            hidden_channels=int(hidden_channels),
            activation="gelu",
            norm=True,
            ch_mults=[1, 2, 2, 4],
            is_attn=[False, False, False, False],
            mid_attn=False,
            n_blocks=2,
            param_conditioning=f"scalar_{int(param_dim)}",
            use_scale_shift_norm=False,
            use1x1=False,
            n_dims=1,
            features="last_layer",
        )

    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # AL4PDE/PDEArena shape: [B, T, C, X] for 1D.
        out = self.model(state[:, None, :, :], z=params, time=None)
        delta = out[:, 0, :, :]
        return state + self.difference_weight * delta
