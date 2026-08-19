# Leonardo CINECA HPC Guide & SLURM Runbook

This runbook documents the environment, layout, and job submission procedures for running large-scale OGAS experiments on the **Leonardo Booster (CINECA)** cluster with NVIDIA A100-SXM4-64GB GPUs.

---

## 1. Filesystem Layout on Leonardo

| Resource | Path | Description |
|---|---|---|
| **Codebase & Configs** | `~/ogas_states` | Git repository and active code |
| **Virtual Environment** | `~/ogas_states/.venv` | Python 3.11 virtualenv with PyTorch cu121 & JAX CUDA |
| **External Solver Clones** | `/leonardo_scratch/fast/EUHPC_D36_033/ogas_external/` | Pinned checkouts of AL4PDE, PDEArena, JAX-CFD |
| **Validation Datasets** | `/leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/` | Seeded, fixed validation banks (Attractor, Hard TV, Tube) |
| **Experiment Output Runs** | `/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/` | Training logs, checkpoints, histories, and metrics |

---

## 2. Environment Setup & Dependencies

```bash
# 1. Navigate to repository
cd ~/ogas_states

# 2. Activate virtualenv
source .venv/bin/activate

# 3. Verify GPU & JAX devices
python -c "import torch, jax; print('PyTorch CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); print('JAX Devices:', jax.devices())"
```

---

## 3. Launching Production SLURM Suites

All cluster jobs are submitted to the `boost_usr_prod` partition with 1 NVIDIA A100 GPU and 8 CPU cores per task:

### A. 1D Kuramoto-Sivashinsky Master Suite
```bash
bash scripts/launch_full_suite_slurm.sh
```
*Launches 10 seeds each for Pushforward, V3 Generative, and Ensemble Scaling ($M \in \{1, 3, 5\}$).*

### B. 1D Viscous Burgers Master Suite
```bash
bash scripts/launch_burgers_suite_slurm.sh
```
*Launches 10 seeds each for Uniform, Heuristic Tube, Sobolev, and Spectrum variants across $M \in \{1, 3, 5\}$.*

### C. 1D Classical Active Learning Baselines
```bash
bash scripts/launch_classical_al_ks_slurm.sh
bash scripts/launch_classical_al_burgers_slurm.sh
```
*Launches Top-K and SBAL ($\alpha=1.0$) candidate acquisition baselines.*

### D. 2D Navier-Stokes Kolmogorov Flow Pilot
```bash
bash scripts/launch_2d_kolmogorov_slurm.sh
```
*Launches 2D AL4PDE CFD benchmarks on $128 \times 128$ resolution.*

---

## 4. Monitoring & Diagnostics

```bash
# Check all running jobs
squeue -u $USER --format="%.10i %.28j %.10T %.10M %.15N"

# Count jobs by state (RUNNING / PENDING)
squeue -u $USER -h -t R,PD | awk '{print $5}' | sort | uniq -c

# Monitor live output log of a specific job
tail -f /leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/<suite>/<run_name>/*.out
```

---

## 5. Post-Processing & Figure Generation

Once jobs are in progress or finished, run the automated analysis pipeline on the login node:

```bash
# 1. Generate all publication figures (PNG / PDF)
python scripts/generate_publication_figures.py

# 2. Generate the comprehensive markdown synthesis
python scripts/generate_full_picture_report.py
```
All outputs are saved to [`docs/figures/`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/figures/) and [`docs/full_picture_report.md`](file:///leonardo/home/userexternal/pcesar00/ogas_states/docs/full_picture_report.md).
