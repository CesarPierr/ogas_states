# AGENT.md: Operational Guide for AI Assistants & Developers

## 1. Project Purpose & Scope

**OGAS** (*Operator-Guided Active Sampling*) is an active learning framework for neural PDE surrogates.
The core hypothesis is that **conditional generative state proposal (OT-CFM / Pushforward)** improves surrogate sample efficiency and multi-step rollout stability by populating an off-attractor recovery tube, outperforming passive solver trajectories and classical active learning.

---

## 2. Invariant Rules & Design Standards

1. **Exact Budget Parity**: Every active learning round $R$ maintains exactly $N = \text{trajectories\_per\_round} \times \text{steps\_per\_trajectory}$ transitions.
2. **Deterministic & Seeded**: All random sampling (parameters, initial conditions, model weights) is seeded and bit-reproducible.
3. **Google / MIT Research Standards**: Keep code concise, explicit, fully typed, with zero unnecessary abstractions or framework bloat.

---

## 3. Key Package Components (`poolbased_surrogate/`)

| Module | Role |
|---|---|
| [`config.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/config.py) | Strongly-typed dataclass YAML configurations (`PDEConfig`, `PoolConfig`, `TrainConfig`). |
| [`pde.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/pde.py) | Unified simulator engine for 1D (KS, Burgers) and 2D (Navier-Stokes Kolmogorov Flow). |
| [`strategies.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/strategies.py) | Acquisition strategy registry: `UniformStrategy`, `HeuristicTubeStrategy`, `ClassicalALStrategy`, `GenerativeStrategy`. |
| [`models/surrogate.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/models/surrogate.py) | `ExactAL4PDEUnet1D`, `ExactAL4PDEUnet2D`, and `EnsembleSurrogate`. |
| [`models/ddpm.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/models/ddpm.py) | OT-CFM Flow Matching (`FlowMatching1D`) and Conditional DDPM (`DDPM1D`). |
| [`eval.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/eval.py) | GPU-vectorized multi-step rollouts and metric evaluation engine. |
| [`run.py`](file:///leonardo/home/userexternal/pcesar00/ogas_states/poolbased_surrogate/run.py) | Active learning pipeline orchestrator. |

---

## 4. Key Execution Commands

### Local Smoke Test (< 30 seconds)
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

### SLURM Master Suite Launchers
```bash
# 1D KS Master Suite (Pushforward, V3, Ensemble Scaling across 10 seeds)
bash scripts/launch_full_suite_slurm.sh

# 1D Burgers Master Suite (Uniform, Tube, Sobolev, Spectrum across 10 seeds)
bash scripts/launch_burgers_suite_slurm.sh

# 1D Classical Active Learning (Top-K / SBAL)
bash scripts/launch_classical_al_ks_slurm.sh
bash scripts/launch_classical_al_burgers_slurm.sh

# 2D Navier-Stokes AL4PDE Pilot Suite
bash scripts/launch_2d_kolmogorov_slurm.sh
```

### Post-Processing & Figures
```bash
# Generate all publication figures
python scripts/generate_publication_figures.py

# Generate consolidated report
python scripts/generate_full_picture_report.py
```
