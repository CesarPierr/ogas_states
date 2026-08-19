# 03. Codebase Architecture & Developer Guide

This document provides a deep architectural walkthrough of the **`poolbased_surrogate`** package, illustrating the complete dataflow through an active learning round, tensor conventions, and design patterns.

---

## 1. High-Level Package Architecture

The codebase is organized into modular, decoupled components adhering to Google and MIT research engineering best practices:

```
                            ┌────────────────────────────────────────────────────────┐
                            │                  EXPERIMENT PIPELINE                   │
                            │                   (run.py orchestrator)                │
                            └────────────────────────────────────────────────────────┘
                                                         │
         ┌───────────────────────┬───────────────────────┼───────────────────────┬───────────────────────┐
         ▼                       ▼                       ▼                       ▼                       ▼
   [ config.py ]            [ pde.py ]            [ strategies.py ]        [ models/ ]             [ eval.py ]
 Typed Dataclasses       Unified PDE Simulator   Acquisition Registry   Surrogates & Gen      100% GPU Rollouts
 (PDE, Pool, Train)      (1D KS, Burgers, NS2D)  (Uniform, Tube, AL)    (Exact Unet1D/2D)     (Zero-Sync VRAM)
```

---

## 2. Step-by-Step Anatomy of an Active Learning Round

An entire active learning experiment proceeds in discrete rounds $r = 0, \dots, R_{\max}$. Here is the precise sequence of operations executed during each round:

```
                        ROUND r = 0 (Initialization)
                        ═════════════════════════════
                        1. Sample initial conditions u_0 and parameters \mu uniformly.
                        2. Simulate full solver trajectories -> flatten to transitions (u_t, \mu, u_{t+1}).
                        3. Train Surrogate Model \mathcal{S}_\theta on \mathcal{D}^{(0)}.
                        4. Evaluate on fixed 4-bank test suite (Attractor, Hard Low-Amp, Hard TV, Tube).
                        5. Compute per-transition surrogate losses \ell_i.
                        6. Train Conditional Generator p_\phi(u | c) on (u_i, c_i).
                                     │
                                     ▼
                        ROUND r >= 1 (Active Cycle)
                        ═══════════════════════════
                        1. Strategy Acquisition (strategies.py):
                           - 50% Uniform Trajectories from Solver.
                           - 50% Generative Proposal from Flow Matching / Classical AL.
                        2. Query Solver: advance proposed states by 1 step: u_{t+1} = \Psi_{\Delta t}(u_t).
                        3. Merge into Active Pool: \mathcal{D}^{(r)} = \mathcal{D}^{(r-1)} \cup \mathcal{D}_{\text{new}}.
                        4. Re-train Surrogate Ensemble \mathcal{S}_\theta on \mathcal{D}^{(r)}.
                        5. Re-evaluate on fixed 4-bank test suite.
                        6. Update per-transition losses & re-train Generator p_\phi.
                        7. Save checkpoint (weights, history.json, pool_round_r.npz).
```

---

## 3. Core Modules Deep Dive

### A. [`poolbased_surrogate/config.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/config.py)
Uses Python dataclasses with full type annotations to validate configurations from YAML:
- `PDEConfig`: PDE name (`ks`, `burgers`, `ns2d`), spatial resolution $N$, time-step $\Delta t$, physical parameter ranges, initial condition spectrum.
- `PoolConfig`: Trajectories per round, steps per trajectory, sampling strategy variant (`uniform_baseline`, `heuristic_tube`, `classic_al_topk`, `classic_al_sbal`, `ogas_generative`).
- `SurrogateConfig`: Model architecture (`al4pde_unet1d`, `al4pde_unet2d`), hidden channel width, depth, ensemble size $M$, training epochs, learning rate.
- `GeneratorConfig`: Model type (`flow_matching`, `ddpm`), quantile proposal distributions, conditioning modes.

---

### B. [`poolbased_surrogate/pde.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/pde.py)
The unified physical simulation engine bridging 1D and 2D solvers with zero boilerplate:
```python
class PDE:
    def __init__(self, cfg: PDEConfig):
        # Automatically detects spatial_dim and initializes AL4PDE/JAX backend
        ...
    
    def sample_params(self, n: int, seed: int) -> np.ndarray:
        """Sample physical PDE parameters (viscosity, domain length) -> [n, P]"""
        
    def sample_ic(self, n: int, seed: int) -> np.ndarray:
        """Sample physical initial conditions -> [n, C, N] (1D) or [n, 2, H, W] (2D)"""
        
    def simulate(self, states: np.ndarray, params: np.ndarray, steps: int) -> np.ndarray:
        """Simulate multi-step trajectories -> [n, steps+1, C, N]"""
        
    def step(self, states: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Advance single-step transitions -> [n, C, N]"""
```

---

### C. [`poolbased_surrogate/strategies.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/strategies.py)
Implements the Strategy Pattern for clean, decoupled data acquisition:
- `UniformStrategy`: Generates 100% full solver trajectories from uniform random initial conditions.
- `HeuristicTubeStrategy`: Applies calibrated smooth spectral noise envelopes around existing attractor trajectories to probe the local recovery basin.
- `ClassicalALStrategy`: Generates $10\times$ candidate trajectories from the solver, scores them via Ensemble Variance ($M \ge 3$) or Residual Loss ($M=1$), and performs Top-K or SBAL selection.
- `GenerativeStrategy`: Uses the trained OT-CFM Flow Matching generator to synthesize off-attractor boundary states conditioned on high loss quantiles, stepping each state once with the solver.

---

### D. [`poolbased_surrogate/models/surrogate.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/models/surrogate.py)
Provides native neural operator architectures with exact parity to the AL4PDE benchmark:
- `ExactAL4PDEUnet1D`: 1D Conditioned U-Net with circular boundary padding and residual scaling.
- `ExactAL4PDEUnet2D`: 2D Conditioned U-Net operating on 2-channel velocity fields $(v_x, v_y)$.
- `EnsembleSurrogate`: Wraps an ensemble of $M$ models, computing forward predictions $\bar{\mathcal{S}}(u, \mu)$ and vectorized epistemic uncertainty $\sigma^2(u, \mu)$ across the ensemble.

---

### E. [`poolbased_surrogate/eval.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/eval.py)
High-performance evaluation engine:
- **Zero-Sync GPU Vectorization**: Multi-step autoregressive rollouts are executed entirely inside GPU VRAM, eliminating CPU-GPU memory transfer bottlenecks.
- **Multi-Metric Suite**: Computes NRMSE, RMSE, Max-Error, and Spectral Energy spectra across all test banks.

---

## 4. Tensor Shape Contracts

| Tensor Type | 1D Physics (KS, Burgers) | 2D Physics (Navier-Stokes) |
|---|---|---|
| **State Tensor** $u$ | `[B, 1, 128]` | `[B, 2, 128, 128]` (vx, vy) |
| **Parameter Tensor** $\mu$ | `[B, 1]` or `[B, 2]` | `[B, 2]` $(\eta, \zeta)$ |
| **Trajectory Tensor** | `[B, steps+1, 1, 128]` | `[B, steps+1, 2, 128, 128]` |
| **Quantile Condition** $c$ | `[B, 1]` | `[B, 1]` |
| **Ensemble Prediction** | `[B, 1, 128]` | `[B, 2, 128, 128]` |
| **Ensemble Variance** | `[B]` | `[B]` |
