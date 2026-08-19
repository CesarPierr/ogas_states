# 04. Benchmarks, PDE Solvers & Evaluation Protocol

This document details the equations, numerical solvers, parameter ranges, and standardized evaluation test banks used throughout the OGAS benchmark suite.

---

## 1. Supported PDEs & Numerical Solvers

### A. 1D Kuramoto-Sivashinsky (KS)
- **Mathematical Form**:
  $$\partial_t u + u \partial_x u + \alpha \partial_x^2 u + \beta \partial_x^4 u = 0, \quad x \in [0, 32], \; t \ge 0$$
- **Physics**: Models flame front instability, fluid film dynamics, and chaotic spatiotemporal turbulence.
- **Parameters**: Domain length $L \in [0.5, 4.0]$, non-linear scale $\nu \in [0.1, 100.0]$.
- **Solver**: Native AL4PDE `ParametricKSJaxSim` (pseudo-spectral integration via JAX with anti-aliasing).
- **Discretization**: $N=128$, $\Delta t = 0.05$.

### B. 1D Viscous Burgers Equation
- **Mathematical Form**:
  $$\partial_t u + u \partial_x u - \nu \partial_x^2 u = 0, \quad x \in [0, 1], \; t \ge 0$$
- **Physics**: Canonical model for non-linear advection, shock steepening, and viscous dissipation.
- **Parameters**: Viscosity $\nu \in [10^{-3}, 10^{-1}]$ (log-scale sampled).
- **Solver**: Native AL4PDE `BurgersSim` (2nd-order Godunov flux limiter).
- **Discretization**: $N=128$, $\Delta t = 0.05$.

### C. 2D Incompressible Navier-Stokes (Kolmogorov Flow)
- **Mathematical Form**:
  $$\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}, \quad \nabla \cdot \mathbf{u} = 0$$
- **Physics**: 2D turbulent vortex interactions driven by sinusoidal Kolmogorov forcing $\mathbf{f} = \sin(k y)\hat{\mathbf{x}}$.
- **Parameters**: Viscosities $\eta, \zeta \in [10^{-4}, 10^{-1}]$ (log-scale sampled).
- **Solver**: Native AL4PDE / PDEBench `CFDSim` (high-order Riemann HLL/HLLC Godunov solver in JAX).
- **Discretization**: Resolution $128 \times 128$, 2-channel velocity $(v_x, v_y)$, $\Delta t = 0.05$.

---

## 2. Standardized Multi-Bank Evaluation Protocol

To rigorously quantify surrogate performance without attractor overfitting, models are evaluated across **4 standardized test banks**:

| Test Bank | Generation Mechanism | Purpose |
|---|---|---|
| **1. Attractor Validation** | 16 seeded trajectories $\times$ 50 steps from standard attractor ICs. | Baseline in-distribution accuracy. |
| **2. Hard Low-Amplitude** | Initial conditions scaled below attractor energy floor ($E < E_{\text{attractor}}$). | Tests linear growth and recovery towards the attractor. |
| **3. Hard Total Variation (High TV)** | Initial conditions with injected high-frequency Fourier modes. | Tests surrogate resistance to non-physical high-frequency blowup. |
| **4. Perturbation Tubes** | Ground-truth trajectories perturbed by smooth spectral envelopes ($\rho \in [0.1, 0.5]$). | Directly measures off-manifold self-correcting stability. |

---

## 3. Standard Metrics Computed

All evaluations compute:
- **1-step & Multi-step NRMSE**: Normalized Root Mean Squared Error $\frac{\| \hat{u} - u \|_2}{\| u \|_2}$.
- **RMSE**: Absolute Root Mean Squared Error.
- **Max Error ($L_\infty$)**: Peak point-wise discrepancy.
- **Spectral Error**: Fourier power spectrum divergence across spatial wavenumbers $k$.
