# 01. Project Overview & Scientific Narrative

## The Core Challenge: Why Neural PDE Surrogates Fail in Deployment

Deep learning surrogates for Partial Differential Equations (e.g., Fourier Neural Operators, U-Net autoregressive models) have emerged as powerful tools for accelerating physical simulations by $100\times$ to $1000\times$. 

However, when deployed in realistic scientific or engineering settings, standard neural surrogates suffer from a fatal flaw: **autoregressive error accumulation and distribution shift**.

```
Training on Attractor:            Deployment (Multi-Step Rollout):
       ────────────────                  ────┐  (step 1: small error)
       (Attractor Tube)                      └───┐  (step 5: off-attractor)
       ────────────────                          └───►  (step 20: DIVERGENCE / BLOWUP)
```

1. **Passive Trajectory Sampling is Narrow**: Training data is traditionally gathered by rolling out numerical solvers from random initial conditions. As the PDE evolves, trajectories rapidly relax onto a low-dimensional **attractor manifold**.
2. **Autoregressive Drift**: During inference, small imperfect predictions at step $t$ act as perturbed initial states for step $t+1$. 
3. **Catastrophic Failure**: Because the surrogate has never seen off-attractor states during training, it cannot recover. High-frequency artifacts and unphysical oscillations rapidly blow up.

---

## Why Standard Active Learning is Not Enough

In machine learning, **Active Learning (AL)** selects the most informative data points to label. In PDE modeling, classical AL typically generates candidate trajectory batches (e.g., $10\times$ oversampling) and selects points with maximal ensemble variance or residual loss.

However, classical AL has severe limitations for dynamical systems:
- **Massive Computational Waste**: Simulating $10\times$ candidate trajectories with high-fidelity solvers consumes enormous compute, only to discard 90% of the calculated states.
- **Attractor Trapping**: Candidate solver trajectories still remain confined to the attractor! Classical AL cannot discover truly off-manifold boundary states that test surrogate robustness.

---

## The OGAS Solution: Generative Active Sampling

**OGAS** (*Operator-Guided Active Sampling*) fundamentally flips the data acquisition paradigm:

> **Instead of waiting for a solver to stumble upon difficult states, we use conditional generative modeling to synthesize challenging, out-of-attractor physical states directly, and query the solver for a single step.**

```
                        ┌───────────────────────────────┐
                        │      OGAS DUAL-POOL LOOP      │
                        └───────────────────────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
      [ Attractor Anchor ]                           [ Generative Exploration ]
     50% Standard Solver                             50% Conditional Flow Matching
    (Maintains natural physics)                      (Probes surrogate weaknesses)
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                         [ Single Transition Dataset ]
                           (u_t, \mu) -> u_{t+1}
                                        │
                                        ▼
                         [ Surrogate & Generator Update ]
```

### Key Innovations of OGAS:
1. **Targeted Difficulty via Quantile Conditioning**:
   The generative model learns $p_\phi(u \mid c)$, where $c \in [0, 1]$ represents the surrogate loss quantile. Sampling at $c \ge 0.8$ produces states tailored to current model weaknesses.
2. **Multi-Step Pushforward Stability**:
   Conditioning on multi-step horizon errors teaches the generator to target states that would otherwise trigger rollout divergence.
3. **Stabilizing "Tube" Geometry**:
   By learning single-step recovery dynamics around the attractor, the surrogate gains self-correcting behavior, dramatically stabilizing long rollouts.
4. **Extreme Sample Efficiency**:
   Zero wasted solver trajectory evaluations: 100% of queried solver transitions enter the training pool.
