# OGAS: Operator-Guided Generative Active Sampling for Neural PDE Surrogates

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg)](https://pytorch.org/)
[![JAX / JAX-CFD](https://img.shields.io/badge/JAX-0.4+-red.svg)](https://github.com/google/jax)
[![AL4PDE Symmetrical](https://img.shields.io/badge/AL4PDE-Parity-green.svg)](https://github.com/tum-pbs/al4pde)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OGAS** (*Operator-Guided Active Sampling*) is an active learning framework designed to train neural PDE surrogates (Neural Operators, Conditioned U-Nets, FNOs) with maximum sample efficiency and rollout stability.

Instead of passively collecting trajectory snapshots along standard PDE solver attractors, OGAS actively probes the state space. It leverages **conditional generative modeling (Optimal Transport Flow Matching / DDPM)** to synthesize challenging, out-of-attractor, and boundary physical states conditioned on surrogate error/uncertainty quantiles, which are then queried via exact numerical solvers.

---

## 📖 Table of Contents
1. [Scientific Motivation & Core Idea](#-scientific-motivation--core-idea)
2. [How OGAS Works: The Active Loop](#-how-ogas-works-the-active-loop)
3. [Supported Equations & Solvers](#-supported-equations--solvers)
4. [Acquisition Strategies](#-acquisition-strategies)
5. [Repository Structure](#-repository-structure)
6. [Quickstart (< 5 minutes)](#-quickstart--5-minutes)
7. [Cluster Deployment on HPC (SLURM / Leonardo Booster)](#-cluster-deployment-on-hpc-slurm--leonardo-booster)
8. [Analysis & Publication Figures](#-analysis--publication-figures)
9. [Key References & Documentation Pointers](#-key-references--documentation-pointers)

---

## 🔬 Scientific Motivation & Core Idea

Neural surrogate models $u_{t+1} \approx \mathcal{S}_\theta(u_t, \mu)$ trained purely on unperturbed numerical solver trajectories suffer from **autoregressive error accumulation**:
1. During long rollouts, small prediction errors push the surrogate outside the narrow manifold of training attractors.
2. Once off-manifold, the surrogate encounters unfamiliar states and quickly diverges (spectral explosion, non-physical shocks).
3. Standard Active Learning (e.g., querying trajectories with highest ensemble variance) is computationally inefficient because it requires rolling out entire candidate trajectories ($10\times$ solver overhead) only to discard most points.

**The OGAS Solution**:
- OGAS trains a **conditional generative model** $p_\phi(u \mid c)$ (where $c \in [0, 1]$ represents the surrogate loss quantile or difficulty target).
- In each round $R$, OGAS samples states directly in high-loss regions of state space, evaluates one single solver step $u_{t+1} = \text{Solver}(u_t, \mu)$, and updates both the surrogate and the generator.
- This creates a **stabilizing "tube" of data** around the attractor, granting robust multi-step rollouts and superior out-of-distribution generalization.

```
       ┌─────────────────────────────────────────────────────────────┐
       │                 OGAS ACTIVE LEARNING ROUND                  │
       └─────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
    [ Uniform Solver Pool ]                       [ Generative Proposal ]
  100% Attractor Trajectories                  Flow Matching / Pushforward
    (50% of round budget)                         (50% of round budget)
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                       [ Unified Transition Pool ]
                       (u_t, \mu) -> u_{t+1}
                                      │
                                      ▼
                       [ Train Surrogate Ensemble ]
                          \mathcal{S}_\theta(u_t, \mu)
                                      │
                                      ▼
                       [ Train Conditional Generator ]
                             p_\phi(u \mid loss)
                                      │
                                      ▼
                       [ 100% GPU Vectorized Eval ]
                   (Attractor, Tube, Sobolev, Rollouts)
```

---

## 🌊 Supported Equations & Solvers

OGAS uses native integrations with **AL4PDE** and **PDEBench** to guarantee 100% architectural parity across 1D and 2D:

| PDE Equation | Dimension | Physics | Parameters ($\mu$) | Solver Backend |
|---|---|---|---|---|
| **Kuramoto-Sivashinsky (KS)** | 1D Periodic | Chaotic spatiotemporal turbulence, flame front flutter | Length $L \in [0.5, 4.0]$, Viscosity $\nu$ | Pseudo-spectral (JAX) |
| **Viscous Burgers** | 1D Periodic | Non-linear advection & shock formation | Viscosity $\nu \in [10^{-3}, 10^{-1}]$ | Finite Difference / Spectral |
| **Navier-Stokes (Kolmogorov)** | 2D Incompressible | 2D Vortex shedding, turbulent energy cascade | Viscosity $\eta, \zeta \in [10^{-4}, 10^{-1}]$ | AL4PDE `CFDSim` (JAX-CFD) |

---

## 🎯 Acquisition Strategies

All strategies are unified under [`poolbased_surrogate/strategies.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/strategies.py):

1. **`uniform_baseline`** : Standard passive sampling. 100% full solver trajectories from uniform initial conditions.
2. **`heuristic_tube`** : 50% Uniform + 50% spectral perturbations around attractor trajectories. Proves that exploring off-manifold state space improves rollouts.
3. **`classic_al_topk` / `classic_al_sbal`** : Classical pool-based Active Learning. Generates $10\times$ solver candidate trajectories, scores them by Ensemble Variance ($M \ge 3$) or Residual Loss ($M=1$), and acquires Top-K or SBAL ($\alpha=1.0$) states.
4. **`ogas_generative` (Flow Matching & Pushforward)** : Generates states conditioned on high loss quantiles via Optimal Transport Conditional Flow Matching (OT-CFM). Pushforward variant multi-step unrolls generated states to anchor long-horizon stability.

---

## 📁 Repository Structure

The codebase is engineered to Google / MIT research standards: lean, modular, typed, and fully reproducible:

```
ogas_states/
├── configs/                      # YAML configuration files
│   ├── phase2_ks_v3.yaml         # Master KS 1D configuration
│   ├── phase2_burgers.yaml       # Master Burgers 1D configuration
│   └── phase2_ns2d.yaml          # Master Navier-Stokes 2D configuration
│
├── poolbased_surrogate/          # Core minimalist Python package (< 1500 LOC)
│   ├── config.py                 # Typed configuration dataclasses
│   ├── pde.py                    # Unified 1D & 2D PDE simulator interface
│   ├── strategies.py             # Acquisition strategy registry (Uniform, Tube, AL, OGAS)
│   ├── models/
│   │   ├── surrogate.py          # ExactAL4PDEUnet1D, ExactAL4PDEUnet2D, EnsembleSurrogate
│   │   └── ddpm.py               # OT-CFM Flow Matching & Conditional DDPM
│   ├── train.py                  # Surrogate & generator training loops
│   ├── eval.py                   # GPU-vectorized evaluation & rollout engine (Zero-Sync)
│   └── run.py                    # Main Active Learning pipeline orchestrator
│
├── scripts/                      # Production SLURM launchers & synthesis tools
│   ├── launch_full_suite_slurm.sh          # Master KS 1D SLURM suite
│   ├── launch_burgers_suite_slurm.sh       # Master Burgers 1D SLURM suite
│   ├── launch_classical_al_ks_slurm.sh     # KS Classical AL baselines
│   ├── launch_classical_al_burgers_slurm.sh# Burgers Classical AL baselines
│   ├── launch_2d_kolmogorov_slurm.sh       # 2D Navier-Stokes AL4PDE suite
│   ├── generate_publication_figures.py     # Master publication figure generator
│   └── generate_full_picture_report.py     # Consolidated Markdown/JSON report generator
│
├── docs/                         # Reports, figures, and technical deep-dives
│   ├── full_picture_report.md    # Master comprehensive scientific synthesis
│   ├── round_evolution.md        # Per-round metric evolution & analysis
│   └── figures/                  # Publication figures (Heatmaps, Rollouts, Spectra)
│
├── pyproject.toml                # Package definition
└── README.md                     # Master documentation (this file)
```

---

## ⚡ Quickstart (< 5 minutes)

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/CesarPierr/ogas_states.git
cd ogas_states

# Create environment with uv or venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2. Run a Fast Local Smoke Test (1 Round)
```bash
python -m poolbased_surrogate.run \
  --config configs/phase2_ks_v3.yaml \
  --output ./test_run \
  --override pool.n_rounds=2 \
  --override pool.trajectories_per_round=4 \
  --override pool.steps_per_trajectory=5 \
  --override surrogate.epochs=2 \
  --override generator.epochs=2 \
  --override wandb.enabled=false
```

---

## 🚀 Cluster Deployment on HPC (SLURM / Leonardo Booster)

To run large-scale benchmarks on GPU clusters (NVIDIA A100 / Leonardo CINECA):

```bash
# Launch KS 1D Master Suite (Pushforward, V3, Ensemble Scaling across 10 seeds)
sbatch scripts/launch_full_suite_slurm.sh

# Launch Burgers 1D Suite (Uniform, Tube, Sobolev, Spectrum across 10 seeds)
sbatch scripts/launch_burgers_suite_slurm.sh

# Launch Classical Active Learning Baselines (Top-K / SBAL)
sbatch scripts/launch_classical_al_ks_slurm.sh
sbatch scripts/launch_classical_al_burgers_slurm.sh

# Launch 2D Navier-Stokes Kolmogorov Flow Suite
sbatch scripts/launch_2d_kolmogorov_slurm.sh
```

---

## 📊 Analysis & Publication Figures

Once runs are complete or in progress, generate all publication figures and quantitative summary tables with a single command:

```bash
# Generate all 10 publication figures in docs/figures/
python scripts/generate_publication_figures.py

# Generate full statistical report in docs/full_picture_report.md
python scripts/generate_full_picture_report.py
```

### Key Figures Generated:
- **`fig1_convergence_iso_m.png`** : Sample efficiency and NRMSE convergence across acquisition rounds.
- **`fig4_rollout_stability.png`** : Autoregressive multi-step rollout RMSE up to 50 steps.
- **`fig8_multimetric_heatmap_gains.png`** : Master cross-metric heatmap comparing all methods against uniform baseline.
- **`fig9_rollout_horizon_time_series.png`** : Temporal error trajectory showing long-term stability of generative sampling.

---

## 📚 Key References & Documentation Pointers

- **Full Scientific Report** : [`docs/full_picture_report.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/full_picture_report.md)
- **Round-by-Round Evolution** : [`docs/round_evolution.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/round_evolution.md)
- **Convergence Efficiency Guide** : [`docs/convergence_efficiency.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/convergence_efficiency.md)
- **HPC & Cluster Setup Guide** : [`LEONARDO.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/LEONARDO.md)
