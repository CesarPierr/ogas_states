#!/bin/bash
#SBATCH --job-name=eval_posthoc
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=euhpc_d36_033
#SBATCH --qos=normal
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/eval_posthoc_%j.out

cd /leonardo/home/userexternal/pcesar00/ogas_states

# Load modules
module load profile/deeplrn 2>/dev/null || true
module load python/3.11.6--gcc--12.2.0 2>/dev/null || true
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

# Set PYTHONPATH
export PYTHONPATH=/leonardo_scratch/fast/EUHPC_D36_033/ogas_external/pdearena:/leonardo_scratch/fast/EUHPC_D36_033/ogas_external/al4pde:$PYTHONPATH
export OMP_NUM_THREADS=16

# Run evaluation
.venv/bin/python scripts/eval_campaign_posthoc_detailed.py \
    --base-dir /leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_phase2_repro \
    --uniform /leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/ks_res800_al4pde_sub5_seed43_n1500_t100.npz \
    --hard /leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/ks_diverse_suite \
    --baselines uniform_baseline \
    --out campaign_detailed_posthoc_analysis_v2.json \
    --device cuda

echo "Evaluation complete."
