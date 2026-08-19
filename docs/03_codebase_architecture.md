# 03. Codebase Architecture & Developer Guide

This document explains the architecture of the **`poolbased_surrogate`** package, tensor conventions, design decisions, and extension points.

---

## 1. Package Organization

The core package is structured into clean, decoupled modules adhering to Google/MIT research standards:

```
poolbased_surrogate/
├── config.py         # Strongly-typed configuration dataclasses
├── pde.py            # Unified 1D & 2D PDE engine (AL4PDE bridge)
├── strategies.py     # Unified active learning acquisition strategies
├── models/
│   ├── surrogate.py  # ExactAL4PDEUnet1D, ExactAL4PDEUnet2D, EnsembleSurrogate
│   └── ddpm.py       # Conditional OT-CFM Flow Matching & DDPM
├── train.py          # Training loops for surrogate & generative models
├── eval.py           # GPU-vectorized rollout and evaluation suite (Zero CPU Sync)
├── pool.py           # Pool transition dataset management facade
└── run.py            # Orchestrator of the active learning loop
```

---

## 2. Tensor Shape Conventions

All tensors across the codebase adhere to strict shape conventions:

| Domain | Spatial Dim | Initial State $u_0$ | Trajectory Tensor | Physical Params $\mu$ |
|---|---|---|---|---|
| **1D PDEs** (KS, Burgers) | $1\text{D}$ ($N=128$) | `[B, 1, N]` | `[B, steps+1, 1, N]` | `[B, P]` |
| **2D PDEs** (Navier-Stokes) | $2\text{D}$ ($128 \times 128$) | `[B, 2, H, W]` (vx, vy) | `[B, steps+1, 2, H, W]` | `[B, P]` |

All arrays in NumPy are `float32`, and all PyTorch models operate on `torch.float32` tensors on `device="cuda"`.

---

## 3. Key Modules & Design Roles

### A. [`poolbased_surrogate/pde.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/pde.py)
Provides a unified simulator interface:
```python
pde = PDE(cfg.pde)
params = pde.sample_params(n, seed)          # [n, P]
states = pde.sample_ic(n, seed)              # [n, C, N] or [n, 2, H, W]
trajs  = pde.simulate(states, params, steps) # [n, steps+1, C, N]
next_u = pde.step(states, params)            # [n, C, N]
```

### B. [`poolbased_surrogate/strategies.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/strategies.py)
Implements the Strategy Pattern for data acquisition:
- `UniformStrategy`: Standard uniform attractor sampling.
- `HeuristicTubeStrategy`: Perturbed spectral noise around attractor.
- `ClassicalALStrategy`: $10\times$ candidate oversampling + Top-K / SBAL selection.
- `GenerativeStrategy`: Conditional Flow Matching proposal.

### C. [`poolbased_surrogate/models/surrogate.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/models/surrogate.py)
Implements exact conditional U-Nets matching the AL4PDE benchmark:
- `ExactAL4PDEUnet1D`: 1D conditioned U-Net with circular padding and difference residual scaling.
- `ExactAL4PDEUnet2D`: 2D conditioned U-Net for velocity fields.
- `EnsembleSurrogate`: Vectorized ensemble wrapper with `.uncertainty(state, params)` method.

### D. [`poolbased_surrogate/eval.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/eval.py)
High-performance evaluation engine:
- **Zero-Sync GPU Rollouts**: Multi-step autoregressive rollouts remain 100% on VRAM without CPU synchronization barriers.
- **Multi-Metric Suite**: Computes NRMSE, RMSE, Max-Error, and Spectral Energy spectra across all test banks.
