# 04. Benchmarks, PDE Solvers & Multi-Bank Evaluation Protocol

This document provides complete physical descriptions, mathematical equations, parameter domains, and verification test banks used across the **OGAS** benchmark suite.

---

## 1. Physical Systems & PDE Solvers

### A. 1D Kuramoto-Sivashinsky (KS) Equation
The Kuramoto-Sivashinsky equation is a canonical non-linear dissipative PDE modeling chaotic spatiotemporal turbulence, flame front flutter, and falling liquid films:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + \alpha \frac{\partial^2 u}{\partial x^2} + \beta \frac{\partial^4 u}{\partial x^4} = 0, \quad x \in [0, 32], \; t \ge 0$$

- **Physical Roles of Terms**:
  - $u \partial_x u$: Non-linear convective energy transfer across spatial scales.
  - $\alpha \partial_x^2 u$: Negative diffusion (energy production at large wavelengths / instability).
  - $\beta \partial_x^4 u$: Fourth-order hyper-viscous dissipation (energy damping at small wavelengths).
- **Chaos & Attractor Dynamics**: The system possesses positive Lyapunov exponents, exhibiting sensitive dependence on initial conditions and a chaotic strange attractor with complex spatio-temporal cell dynamics.
- **Discretization & Solver**:
  - Resolution: $N = 128$ grid points.
  - Solver: Native AL4PDE `ParametricKSJaxSim` (pseudo-spectral Fourier integration with 2/3 de-aliasing rule).
  - Time-step: $\Delta t = 0.05$.
  - Parameter Range: Domain length scale $L \in [0.5, 4.0]$, non-linear scale $\nu \in [0.1, 100.0]$.

---

### B. 1D Viscous Burgers Equation
The Viscous Burgers equation is a fundamental non-linear conservation law modeling shock wave development, acoustic turbulence, and gas dynamics:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0, \quad x \in [0, 1], \; t \ge 0$$

- **Physical Roles of Terms**:
  - $u \partial_x u$: Non-linear self-advection causing wave steepening and shock formation.
  - $\nu \partial_x^2 u$: Physical viscosity smoothing steep gradients and dissipating shock energy.
- **Discretization & Solver**:
  - Resolution: $N = 128$ grid points.
  - Solver: Native AL4PDE `BurgersSim` (2nd-order Godunov finite volume solver with Van Leer flux limiter).
  - Time-step: $\Delta t = 0.05$.
  - Parameter Range: Kinematic viscosity $\nu \in [10^{-3}, 10^{-1}]$ sampled in logarithmic space.

---

### C. 2D Incompressible Navier-Stokes (Kolmogorov Flow)
The 2D Navier-Stokes equations under sinusoidal Kolmogorov forcing describe 2D turbulent vortex decay and energy cascade:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}, \quad \nabla \cdot \mathbf{u} = 0$$

where $\mathbf{f}(x, y) = \sin(k y) \hat{\mathbf{x}}$ is the external Kolmogorov force driving the flow.
- **Physical Roles of Terms**:
  - $(\mathbf{u} \cdot \nabla)\mathbf{u}$: Non-linear vortex stretching and advection.
  - $\nu \nabla^2 \mathbf{u}$: Viscous vortex dissipation.
  - $\mathbf{f}$: Energy injection maintaining chaotic vortex shedding.
- **Discretization & Solver**:
  - Resolution: $128 \times 128$ spatial grid.
  - Channels: 2-channel velocity field $\mathbf{u} = (v_x, v_y)$.
  - Solver: AL4PDE / PDEBench `CFDSim` (Riemann HLL/HLLC Godunov solver implemented in JAX-CFD).
  - Time-step: $\Delta t = 0.05$.
  - Parameter Range: Shear and bulk viscosities $\eta, \zeta \in [10^{-4}, 10^{-1}]$ sampled in logarithmic space.

---

## 2. Multi-Bank Evaluation Test Suites

Evaluating surrogates only on random attractor trajectories creates an illusion of generalization. To rigorously assess surrogate performance, OGAS benchmarks every model across **4 standardized, fixed test banks**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │               STANDARDIZED TEST SUITES                 │
                       └────────────────────────────────────────────────────────┘
                                                    │
         ┌───────────────────┬──────────────────────┴──────────────────────┬───────────────────┐
         ▼                   ▼                                             ▼                   ▼
  [ Bank 1: Attractor ] [ Bank 2: Low-Amp ]                         [ Bank 3: High TV ]  [ Bank 4: Tubes ]
  16 In-Dist Trajs      Energy < E_attractor                        Injected High Modes   Off-Manifold Shocks
  (Attractor Accuracy)  (Linear Recovery)                           (High-Freq Stability)(Self-Correction)
```

### Bank 1: Attractor In-Distribution Validation
- **Protocol**: 16 fixed trajectories $\times$ 50 steps initialized from standard random Fourier modes.
- **Purpose**: Quantifies standard in-distribution surrogate accuracy on the unperturbed physical manifold.

### Bank 2: Hard Low-Amplitude (Out-of-Distribution)
- **Protocol**: 16 fixed trajectories initialized with energy scaled below the attractor floor ($E < 0.1 \cdot E_{\text{attractor}}$).
- **Purpose**: Tests whether the surrogate can accurately capture linear growth regimes and recover the non-linear attractor from non-equilibrium states.

### Bank 3: Hard High Total Variation (High TV)
- **Protocol**: 16 fixed trajectories initialized with high-frequency spatial modes.
- **Purpose**: Tests surrogate resistance against high-frequency numerical blowup and unphysical oscillations.

### Bank 4: Perturbation Tubes ($\rho \in [0.1, 0.5]$)
- **Protocol**: Ground-truth attractor trajectories perturbed by smooth spectral noise envelopes with relative amplitude $\rho$.
- **Purpose**: Directly evaluates the surrogate's off-manifold **contractive restoring force**—the core capability unlocked by OGAS.

---

## 3. Quantitative Metrics Computed

Across all test banks, evaluations record:
1. **Normalized Root Mean Squared Error (NRMSE)**:
   $$\text{NRMSE}(t) = \frac{\| \hat{u}(t) - u(t) \|_{L^2}}{\| u(t) \|_{L^2}}$$
2. **Mean Multi-Step Rollout Error**:
   $$\text{Mean Rollout RMSE} = \frac{1}{K} \sum_{k=1}^K \| \hat{u}_k - u_k \|_{L^2}$$
3. **Peak Pointwise Error ($L_\infty$)**:
   $$\text{Max Error} = \max_{x \in \Omega} |\hat{u}(x, t) - u(x, t)|$$
4. **Spectral Divergence**:
   $$\Delta E(k) = |\mathcal{F}[\hat{u}](k)|^2 - |\mathcal{F}[u](k)|^2$$
   measuring energy conservation across spatial frequencies $k$.
