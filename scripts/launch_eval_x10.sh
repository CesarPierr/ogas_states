#!/bin/bash
#SBATCH --job-name=eval_x10
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=euhpc_d36_033
#SBATCH --qos=normal
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/eval_x10_%j.out

cd /leonardo/home/userexternal/pcesar00/ogas_states

# Load modules
module load profile/deeplrn 2>/dev/null || true
module load python/3.11.6--gcc--12.2.0 2>/dev/null || true
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

# Set PYTHONPATH
export PYTHONPATH=/leonardo_scratch/fast/EUHPC_D36_033/ogas_external/pdearena:/leonardo_scratch/fast/EUHPC_D36_033/ogas_external/al4pde:$PYTHONPATH
export OMP_NUM_THREADS=8

TARGET_DIR="${1:-/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_ensemble_scaling}"
OUT_1STEP="${2:-ensemble_scaling_analysis_x10.json}"
OUT_ROLLOUT="${3:-ensemble_scaling_rollout_x10.json}"

UNIFORM_X10="/leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/ks_res800_al4pde_sub5_seedALL_n15000_t100.npz"
HARD_X10="/leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/ks_diverse_suite_x10"

echo "=== 1. Running 1-step evaluation on high-statistics x10 suite ==="
.venv/bin/python scripts/eval_campaign_posthoc_detailed.py \
    --base-dir "$TARGET_DIR" \
    --uniform "$UNIFORM_X10" \
    --hard "$HARD_X10" \
    --baselines ens5_uniform_baseline ens3_uniform_baseline ens1_uniform_baseline uniform_baseline \
    --out "$OUT_1STEP" \
    --device cuda

echo "=== 2. Running rollout evaluation on high-statistics x10 suite ==="
.venv/bin/python scripts/eval_rollout_posthoc.py \
    --runs "$TARGET_DIR"/* \
    --uniform "$UNIFORM_X10" \
    --steps 100 \
    --baseline ens5_uniform_baseline \
    --out "$OUT_ROLLOUT" \
    --device cuda

echo "High-statistics evaluation complete!"
