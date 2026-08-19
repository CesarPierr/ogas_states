# HANDOFF.md: Project State & Research Context

## 1. Project Context & Current State

The **OGAS** (*Operator-Guided Active Sampling*) codebase has undergone a complete architectural refactoring:
- **Zero Dead Code**: All legacy files, non-Leonardo scripts, and outdated prototypes have been purged (-105k LOC).
- **1D $\leftrightarrow$ 2D AL4PDE Parity**: Native symmetrical bridges for 1D (Burgers, Kuramoto-Sivashinsky) and 2D (Navier-Stokes Kolmogorov Flow).
- **Clean Strategy Pattern**: Unified `SamplingStrategy` registry (`UniformStrategy`, `HeuristicTubeStrategy`, `ClassicalALStrategy`, `GenerativeStrategy`).
- **GPU-Vectorized Evaluation**: 100% in-VRAM rollouts without CPU-GPU synchronization bottlenecks.

---

## 2. Active Cluster Campaigns on Leonardo Booster

Currently, active SLURM jobs are running across the Leonardo Booster partition:
1. **1D KS Master Suite** : Pushforward ($H=5, 10, 20$), V3 Generative, and Ensemble Scaling ($M \in \{1, 3, 5\}$).
2. **1D KS Classical AL** : Top-K and SBAL ($\alpha=1.0$) across $M \in \{1, 3, 5\}$ and 10 seeds.
3. **1D Burgers Master Suite** : Uniform, Tube, Sobolev, Spectrum across $M \in \{1, 3, 5\}$ and 10 seeds.
4. **1D Burgers Classical AL** : Top-K and SBAL across $M \in \{1, 3, 5\}$ and 10 seeds.
5. **2D Navier-Stokes AL4PDE Pilot** : 5 pilot scenarios on $128 \times 128$ resolution.

---

## 3. Documentation Structure

- [`README.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/README.md) : Master overview, scientific motivation, quickstart, and pointers.
- [`docs/01_project_overview.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/01_project_overview.md) : The scientific narrative and challenge of autoregressive rollout drift.
- [`docs/02_mathematical_formulation.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/02_mathematical_formulation.md) : Mathematical formulation of Flow Matching, quantile conditioning, and Pushforward.
- [`docs/03_codebase_architecture.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/03_codebase_architecture.md) : Developer guide, package layout, and tensor contracts.
- [`docs/04_benchmarks_and_pdes.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/04_benchmarks_and_pdes.md) : Physics of KS, Burgers, Navier-Stokes, and the 4 validation suites.
- [`docs/full_picture_report.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/full_picture_report.md) : Full cross-metric quantitative synthesis.
- [`LEONARDO.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/LEONARDO.md) : Leonardo CINECA HPC runbook.

---

## 4. Immediate Next Steps

1. **Monitor Cluster Execution**: Wait for 1D and 2D jobs to complete on Leonardo Booster.
2. **Execute Final Synthesis**: Run `python scripts/generate_publication_figures.py` and `python scripts/generate_full_picture_report.py` once all runs finish.
