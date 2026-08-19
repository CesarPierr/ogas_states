# OGAS: Scientific Foundations & Mathematical Architecture

This document provides the mathematical formulation, algorithmic specifications, and empirical validation protocol underpinning the **OGAS** (*Operator-Guided Active Sampling*) framework.

---

## 1. Problem Formulation: Active Learning for Neural PDE Surrogates

Consider a parametric time-dependent partial differential equation:
$$\partial_t u(x, t) = \mathcal{N}_\mu[u(x, t)], \quad x \in \Omega, \; t \ge 0$$
where $\mu \in \mathcal{P} \subset \mathbb{R}^d$ denotes physical parameters (e.g., viscosity, domain scale), and $\Omega$ is the spatial domain with periodic boundary conditions.

A numerical solver provides ground-truth transitions:
$$u_{t+\Delta t} = \Psi_{\Delta t}(u_t; \mu)$$

We train a neural surrogate operator $\mathcal{S}_\theta(u_t, \mu)$ parameterized by weights $\theta$ to approximate the one-step forward map $\Psi_{\Delta t}$.

### The Multi-Step Rollout Instability
In practical deployment, the surrogate is applied autoregressively over $K$ steps:
$$\hat{u}_k = \mathcal{S}_\theta(\hat{u}_{k-1}, \mu), \quad \hat{u}_0 = u_0$$

When trained exclusively on attractor trajectory distributions $p_{\text{attractor}}(u)$, small non-zero one-step errors $e_k = \hat{u}_k - u_k$ compound. This induces **covariate shift**, driving $\hat{u}_k$ out of the training distribution into unsupported state space, triggering catastrophic rollout divergence (non-physical high-frequency oscillations or shock blowups).

---

## 2. The OGAS Method: Generative State Acquisition

To robustify $\mathcal{S}_\theta$ against rollout perturbations without exploding the computational solver budget, OGAS maintains a hybrid transition pool across active learning rounds $r = 0, \dots, R$:

$$\mathcal{D}^{(r)} = \mathcal{D}_{\text{unif}}^{(r)} \cup \mathcal{D}_{\text{gen}}^{(r)}$$

- **$\mathcal{D}_{\text{unif}}^{(r)}$ (50% budget)**: Preserves attractor coverage via standard trajectory rollout from random initial conditions.
- **$\mathcal{D}_{\text{gen}}^{(r)}$ (50% budget)**: Samples boundary and high-error states synthesized by a conditional generative model, stepping each state exactly once with the solver.

```
       ┌─────────────────────────────────────────────────────────────┐
       │                   ACTIVE SAMPLING PIPELINE                  │
       └─────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     [ Attractor Manifold ]                        [ Off-Manifold Tube ]
    Uniform Trajectory Solver                   Conditional Flow Matching
         (p_attractor)                             p_\phi(u | loss=q)
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                      [ Transition Pool: (u, \mu, u') ]
                                      │
                                      ▼
                      [ Surrogate Training & Rollout ]
                          \mathcal{S}_\theta(u, \mu)
```

---

## 3. Generative Modeling Engines

### A. Optimal Transport Conditional Flow Matching (OT-CFM)
We model vector fields $v_t(u \mid c)$ that transport a standard Gaussian prior $p_0 = \mathcal{N}(0, I)$ to the target data distribution $p_1(u \mid c)$ conditioned on surrogate error/quantile $c \in [0, 1]$.

Under straight probability paths:
$$\psi_t(u_0, u_1) = (1 - t)u_0 + t u_1, \quad t \in [0, 1]$$
the target velocity is $v_t(\psi_t(u_0, u_1)) = u_1 - u_0$.

The CFM objective is minimized via:
$$\mathcal{L}_{\text{CFM}}(\phi) = \mathbb{E}_{t \sim \mathcal{U}(0,1), u_0 \sim p_0, u_1 \sim p_1, c} \left[ \| v_\phi(\psi_t(u_0, u_1), t \mid c) - (u_1 - u_0) \|^2 \right]$$

### B. Loss & Quantile Conditioning
To sample states of controlled difficulty, each transition $(u_i, \mu_i)$ in the active pool is annotated with its realization loss:
$$\ell_i = \| \mathcal{S}_\theta(u_i, \mu_i) - u_i' \|^2$$
We map $\ell_i$ to quantile targets $c_i \in [0, 1]$. In sampling mode, conditioning on $c \ge 0.8$ directs the generative model to synthesize difficult, boundary states where the surrogate is currently weakest.

### C. Multi-Step Pushforward Conditioning
To explicitly combat multi-step rollout drift, the Pushforward variant conditions the generator on multi-step horizon errors:
$$\ell_i^{(H)} = \sum_{k=1}^H \gamma^{k-1} \| \hat{u}_k - u_k \|^2$$
teaching the generator to populate states that directly trigger multi-step divergence.

---

## 4. Benchmark PDEs & Verification Protocol

### A. 1D Kuramoto-Sivashinsky (KS)
- **Equation**: $\partial_t u + u \partial_x u + \alpha \partial_x^2 u + \beta \partial_x^4 u = 0$
- **Properties**: Non-linear energy transfer, chaotic attractor, sensitive spatial modes.
- **Resolution**: $N=128$, domain $L=32.0$, $\Delta t = 0.05$.

### B. 1D Viscous Burgers
- **Equation**: $\partial_t u + u \partial_x u - \nu \partial_x^2 u = 0$
- **Properties**: Advective steepening and viscous shock dissipation.
- **Resolution**: $N=128$, viscosity $\nu \in [10^{-3}, 10^{-1}]$, $\Delta t = 0.05$.

### C. 2D Incompressible Navier-Stokes (Kolmogorov Flow)
- **Equation**: $\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$
- **Properties**: 2D turbulent vortex decay, non-linear Reynolds stress.
- **Resolution**: $128 \times 128$, velocity field $(v_x, v_y)$, solver: AL4PDE `CFDSim`.

---

## 5. Quantitative Evaluation Suites

To prevent overfitting to simple validation trajectories, all algorithms are benchmarked on **4 distinct test suites**:

1. **Attractor Validation**: Fixed 16 trajectories $\times$ 50 steps from standard attractor initial conditions.
2. **Hard Low-Amplitude (Out-of-Distribution)**: Initial conditions with energy below attractor floor, testing linear recovery.
3. **Hard Total Variation (High TV)**: Initial conditions with elevated high-frequency gradients.
4. **Perturbation Tubes ($\rho \in [0.1, 0.5]$)**: Spectral noise envelopes around ground-truth trajectories evaluating off-manifold recovery.
