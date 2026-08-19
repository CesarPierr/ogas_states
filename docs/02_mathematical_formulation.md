# 02. Mathematical Formulation & Algorithmic Foundations

This document provides a comprehensive mathematical treatment of the **OGAS** framework, deriving the continuous normalizing flow objectives, quantile loss conditioning mechanisms, multi-step pushforward formulations, and spectral regularizations.

---

## 1. Problem Setting: Dynamical Systems and Operator Learning

Let $\mathcal{H} = L^2(\Omega; \mathbb{R}^d)$ be a Hilbert space of square-integrable physical fields over a spatial domain $\Omega \subset \mathbb{R}^n$ with periodic boundary conditions. We consider parameterized evolutionary partial differential equations:

$$\frac{\partial u}{\partial t}(x, t) = \mathcal{N}_\mu[u(x, t)], \quad x \in \Omega, \; t \ge 0, \quad u(\cdot, 0) = u_0 \in \mathcal{H}$$

where $\mathcal{N}_\mu: \mathcal{D}(\mathcal{N}_\mu) \subset \mathcal{H} \to \mathcal{H}$ is a non-linear differential operator parameterized by physical constants $\mu \in \mathcal{P} \subset \mathbb{R}^p$ (such as kinematic viscosity $\nu$, domain scale $L$, or external forcing parameters).

### The Continuous Flow Map and Discrete Stepper
The continuous evolution defines a non-linear semigroup (or flow map) $\Phi_t^\mu: \mathcal{H} \to \mathcal{H}$ such that $u(t) = \Phi_t^\mu(u_0)$.

A numerical solver with time-step $\Delta t > 0$ provides a discrete one-step forward operator:
$$\Psi_{\Delta t}(\cdot; \mu): \mathcal{H} \to \mathcal{H}, \quad u_{t+\Delta t} = \Psi_{\Delta t}(u_t; \mu)$$

We seek to train a neural surrogate operator $\mathcal{S}_\theta: \mathcal{H} \times \mathcal{P} \to \mathcal{H}$ parameterized by weights $\theta \in \Theta$ that minimizes the long-horizon autoregressive rollout risk over horizon $K \in \mathbb{N}$:

$$\mathcal{R}_{\text{rollout}}(\theta) = \mathbb{E}_{u_0 \sim p_{\text{IC}}, \, \mu \sim p_\mu} \left[ \frac{1}{K} \sum_{k=1}^K \frac{\| \hat{u}_k - u_k \|_{\mathcal{H}}^2}{\| u_k \|_{\mathcal{H}}^2} \right]$$

where $\hat{u}_k = \mathcal{S}_\theta(\hat{u}_{k-1}, \mu)$ with $\hat{u}_0 = u_0$, and $u_k = (\Psi_{\Delta t})^k(u_0; \mu)$.

---

## 2. Generative Modeling via Optimal Transport Flow Matching (OT-CFM)

Rather than using traditional discrete-time diffusion models (DDPM), which require hundreds of sequential denoising steps, OGAS uses **Continuous Normalizing Flows (CNFs)** parameterized through **Optimal Transport Conditional Flow Matching (OT-CFM)**.

```
       Prior Noise: u_0 ~ N(0, I)                     Target Physical State: u_1 ~ p_data(u | c)
             ┌──────────┐                                      ┌──────────┐
             │          │ ─────────── Straight Path ──────────►│          │
             └──────────┘         \psi_t(u_0, u_1)             └──────────┘
                                         │
                                         ▼
                            Learned Vector Field: v_\phi(u_t, t | c)
```

### A. Straight Probability Paths
Let $p_0 = \mathcal{N}(0, I)$ be the standard Gaussian base density on $\mathcal{H}$, and let $p_1(u \mid c)$ be the conditional target density of physical states given difficulty condition $c \in [0, 1]$.

Under Optimal Transport displacement interpolation, the time-dependent state path between a sampled noise vector $u_0 \sim p_0$ and a physical state $u_1 \sim p_1(\cdot \mid c)$ is defined linearly as:

$$\psi_t(u_0, u_1) = (1 - t) u_0 + t u_1, \quad t \in [0, 1]$$

The instantaneous time-derivative (velocity) along this trajectory is constant:

$$\frac{d}{dt} \psi_t(u_0, u_1) = u_1 - u_0$$

### B. The Flow Matching Objective
We parameterize a time-dependent neural vector field $v_\phi: \mathcal{H} \times [0, 1] \times [0, 1] \to \mathcal{H}$ with weights $\phi$. The Conditional Flow Matching loss is given by:

$$\mathcal{L}_{\text{CFM}}(\phi) = \mathbb{E}_{t \sim \mathcal{U}(0, 1), \, u_0 \sim \mathcal{N}(0, I), \, (u_1, c) \sim \mathcal{D}_{\text{pool}}} \left[ \| v_\phi(\psi_t(u_0, u_1), t \mid c) - (u_1 - u_0) \|_{\mathcal{H}}^2 \right]$$

### C. State Generation via ODE Integration
Once trained, generating a physical state conditioned on difficulty target $c^* \in [0, 1]$ consists of drawing $u_0 \sim \mathcal{N}(0, I)$ and solving the initial value problem:

$$\frac{du(t)}{dt} = v_\phi(u(t), t \mid c^*), \quad u(0) = u_0$$

from $t = 0$ to $t = 1$ using an adaptive Runge-Kutta 4th order (RK4) or Midpoint numerical ODE integrator:

$$u_1 = u_0 + \int_0^1 v_\phi(u(t), t \mid c^*) \, dt$$

---

## 3. Loss & Quantile Conditioning Mechanics

To guide the generative model toward states that challenge the surrogate, each transition in the pool $\mathcal{D}^{(r)}$ is labeled with its current surrogate reconstruction error:

$$\ell_i = \ell(\mathcal{S}_\theta(u_i, \mu_i), u_{i}') = \frac{\| \mathcal{S}_\theta(u_i, \mu_i) - u_{i}' \|_{\mathcal{H}}^2}{\| u_{i}' \|_{\mathcal{H}}^2}$$

### A. Empirical Quantile Transformation
Because absolute loss values scale dynamically across active learning rounds (decreasing as the surrogate improves), raw losses $\ell_i$ cannot be used directly as stationary conditioning variables.

Instead, OGAS maps raw losses to **normalized empirical quantiles**:

$$c_i = F_\ell(\ell_i) = \frac{1}{|\mathcal{D}^{(r)}|} \sum_{j=1}^{|\mathcal{D}^{(r)}|} \mathbb{I}(\ell_j \le \ell_i) \in [0, 1]$$

- $c_i \in [0.0, 0.3]$: Easy attractor transitions (well-resolved smooth flow).
- $c_i \in [0.4, 0.7]$: Intermediate transitions (moderate gradients).
- $c_i \in [0.8, 1.0]$: Hard boundary transitions (steep gradients, near-divergence states).

### B. Adaptive Proposal Distribution
During the data acquisition phase of round $r+1$, generative proposals are drawn from a targeted quantile proposal distribution $p_{\text{prop}}(c)$:

$$c^* \sim \text{Beta}(\alpha=5, \beta=1) \quad \implies \quad \mathbb{E}[c^*] = \frac{5}{6} \approx 0.833$$

This concentrates generative queries in the top $20\%$ hardest difficulty regimes.

---

## 4. Multi-Step Pushforward Formulation

Single-step prediction errors do not always correlate with multi-step autoregressive stability. A surrogate may exhibit low 1-step error while accumulating subtle phase shifts that destabilize rollouts at step 20.

To solve this, the **Pushforward** variant unrolls the surrogate for an autoregressive horizon $H \in \{5, 10, 20\}$:

$$\hat{u}_0 = u_i, \quad \hat{u}_k = \mathcal{S}_\theta(\hat{u}_{k-1}, \mu_i), \quad k = 1, \dots, H$$

The multi-step discounted rollout loss is defined as:

$$\ell_i^{(H)} = \sum_{k=1}^H \gamma^{k-1} \frac{\| \hat{u}_k - (\Psi_{\Delta t})^k(u_i; \mu_i) \|_{\mathcal{H}}^2}{\| (\Psi_{\Delta t})^k(u_i; \mu_i) \|_{\mathcal{H}}^2}$$

where $\gamma \in (0, 1]$ is a temporal discount factor (default $\gamma = 0.95$).

Conditioning the generative model on $c_i^{(H)} = F(\ell_i^{(H)})$ explicitly instructs the generator to synthesize states that act as "seeds" of long-horizon divergence.

---

## 5. Spectral & Sobolev Smoothness Constraints (Sobolev-TV)

To prevent the generative model from synthesizing unphysical high-frequency white noise that does not belong to the PDE's Sobolev space $\mathcal{H}^s(\Omega)$, OGAS incorporates a spectral regularizer.

For a 1D periodic state $u(x)$, let $\hat{u}(k) = \mathcal{F}[u](k)$ denote its discrete Fourier transform. The $H^s$ Sobolev norm is:

$$\| u \|_{H^s}^2 = \sum_{k \in \mathbb{Z}} \left( 1 + |k|^2 \right)^s |\hat{u}(k)|^2$$

In the Sobolev-TV strategy variant, generative samples are filtered or penalized to ensure:

$$\| u_{\text{gen}} \|_{H^s} \le C_{\text{attractor}} \cdot \sup_{u \in \mathcal{A}} \| u \|_{H^s}$$

where $s = 2$ for Kuramoto-Sivashinsky and $s = 1$ for Viscous Burgers. This ensures that generated boundary states retain the sharp but smooth gradient structure of true physical solutions.
