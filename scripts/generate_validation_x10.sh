#!/usr/bin/env bash
#SBATCH --account=EUHPC_D36_033
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=val-gen-x10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/logs/gen_x10_%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/logs/gen_x10_%A_%a.err

set -euo pipefail

ROOT="/leonardo/home/userexternal/pcesar00/ogas_states"
cd "$ROOT"

mkdir -p /leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/logs
mkdir -p /leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/x10_chunks

# On décale la seed de base (43) avec l'index de l'array
SEED=$((43 + SLURM_ARRAY_TASK_ID))
OUT_FILE="/leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/x10_chunks/ks_res800_al4pde_sub5_n1500_t100_chunk${SLURM_ARRAY_TASK_ID}.npz"

echo "Running chunk ${SLURM_ARRAY_TASK_ID} with seed ${SEED} -> ${OUT_FILE}"

export JAX_PLATFORMS=cpu
export PYTHONPATH="/leonardo_scratch/fast/EUHPC_D36_033/ogas_external/pdearena:/leonardo_scratch/fast/EUHPC_D36_033/ogas_external/al4pde:${PYTHONPATH:-}"

./.venv/bin/python scripts/generate_validation_sub5.py \
    --seed $SEED \
    --n-trajectories 1500 \
    --trajectory-steps 100 \
    --resolution 800 \
    --domain-length 64.0 \
    --dt 0.05 \
    --n-substeps 5 \
    --n-warmup 20 \
    --batch-size 50 \
    --output "$OUT_FILE"

echo "Chunk ${SLURM_ARRAY_TASK_ID} complete!"
