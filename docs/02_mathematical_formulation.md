# 02. Mathematical Formulation & Algorithmic Details

This document provides the formal mathematical grounding of the **OGAS** active learning loop, conditional generative engines, and stability regularizations.

---

## 1. Problem Formulation

Let $\Omega \subset \mathbb{R}^d$ be a spatial domain with periodic boundary conditions. We consider parameterized time-dependent PDEs:
$$\partial_t u(x, t) = \mathcal{N}_\mu[u(x, t)], \quad x \in \Omega, \; t \ge 0$$
where $\mu \in \mathcal{P}$ represents physical parameters (viscosity, domain length, forcing frequencies).

A numerical solver provides ground-truth discrete-time transitions:
$$u_{t+\Delta t} = \Psi_{\Delta t}(u_t; \mu)$$

Our goal is to learn a neural surrogate $\mathcal{S}_\theta(u, \mu) \approx \Psi_{\Delta t}(u; \mu)$ that minimizes long-horizon rollout error:
$$\mathcal{L}_{\text{rollout}}(\theta) = \mathbb{E}_{u_0, \mu} \left[ \frac{1}{K} \sum_{k=1}^K \| \hat{u}_k - u_k \|^2 \right]$$
where $\hat{u}_k = \mathcal{S}_\theta(\hat{u}_{k-1}, \mu)$ with $\hat{u}_0 = u_0$.

---

## 2. Conditional Optimal Transport Flow Matching (OT-CFM)

To synthesize states of controlled difficulty, OGAS uses **Continuous Normalizing Flows** via Flow Matching.

### Straight Probability Paths
Given a Gaussian noise prior $u_0 \sim p_0 = \mathcal{N}(0, I)$ and a target physical state $u_1 \sim p_1(u \mid c)$ conditioned on difficulty target $c \in [0, 1]$, we define the time-dependent interpolation:
$$\psi_t(u_0, u_1) = (1 - t)u_0 + t u_1, \quad t \in [0, 1]$$

The analytical velocity vector field generating this path is simply:
$$v_t(\psi_t(u_0, u_1)) = u_1 - u_0$$

### Flow Matching Objective
We parameterize a neural vector field $v_\phi(u, t \mid c)$ (e.g., via a time- and condition-embedded U-Net) and train it by regressing against the straight path velocity:
$$\mathcal{L}_{\text{CFM}}(\phi) = \mathbb{E}_{t \sim \mathcal{U}(0,1), \, u_0 \sim p_0, \, u_1 \sim \mathcal{D}_{\text{pool}}, \, c} \left[ \| v_\phi(\psi_t(u_0, u_1), t \mid c) - (u_1 - u_0) \|^2 \right]$$

### Sampling via ODE Integration
At sampling time, given a requested difficulty quantile $c^* \in [0.8, 1.0]$:
1. Sample prior noise $u_0 \sim \mathcal{N}(0, I)$.
2. Integrate the learned ODE from $t=0$ to $t=1$:
   $$u_1 = u_0 + \int_0^1 v_\phi(u_t, t \mid c^*) \, dt$$
   (implemented via Midpoint or Runge-Kutta 4th-order ODE solvers).

---

## 3. Loss & Quantile Conditioning Mechanics

At each active learning round $r$, transitions $(u_i, \mu_i, u_i')$ in the pool $\mathcal{D}^{(r)}$ are evaluated by the surrogate:
$$\ell_i = \| \mathcal{S}_\theta(u_i, \mu_i) - u_i' \|^2$$

We compute the empirical cumulative distribution function (CDF):
$$c_i = F_\ell(\ell_i) = \frac{1}{|\mathcal{D}|} \sum_{j} \mathbb{I}(\ell_j \le \ell_i) \in [0, 1]$$

- $c_i \approx 0.0$ corresponds to easy, well-fit attractor transitions.
- $c_i \approx 1.0$ corresponds to high-loss, boundary, or sharp-gradient transitions.

---

## 4. Multi-Step Pushforward Rollout Conditioning

In the Pushforward variant, difficulty is not measured purely on single-step error, but on multi-step horizon divergence:
$$\ell_i^{(H)} = \sum_{k=1}^H \gamma^{k-1} \| \hat{u}_k - u_k \|^2$$
where $\hat{u}_k$ is the $k$-step autoregressive rollout initialized at $u_i$, and $\gamma \in (0, 1]$ is a discount factor.

This exposes the generator directly to states that seed multi-step instability, guiding generative sampling to shore up rollout stability.

---

## 5. Frequency & Sobolev Regularization (Sobolev-TV)

To ensure that generated states respect the physical smoothness of the underlying PDE manifold, we optionally add a Sobolev regularizer:
$$\mathcal{L}_{\text{Sobolev}}(u) = \sum_{k} (1 + |k|^2)^s |\hat{u}(k)|^2$$
where $\hat{u}(k)$ are Fourier coefficients and $s \ge 1$. This penalizes unphysical high-frequency noise while preserving valid physical shocks.
