#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ns2d_scaled_sweep"
mkdir -p "$BASE_DIR"

SCENARIOS=(
    "uniform_baseline"
    "heuristic_tube"
    "classic_al_topk"
    "classic_al_sbal"
    "ogas_generative"
)

SEEDS=(101 202 303 404 505)

echo "============================================================================"
echo "=== LAUNCHING 2D SCALED NAVIER-STOKES SWEEP (5 STRATEGIES x 5 SEEDS = 25 JOBS) ==="
echo "============================================================================"

for seed in "${SEEDS[@]}"; do
    for strat in "${SCENARIOS[@]}"; do
        run_dir="${BASE_DIR}/${strat}_seed${seed}"
        mkdir -p "$run_dir"
        
        # Check if already done
        if [ -f "$run_dir/history.json" ]; then
            echo "  [SKIP - ALREADY DONE] ${strat}_seed${seed}"
            continue
        fi
        
        echo "  [SUBMIT] 2D NS Scaled: strategy=${strat}, seed=${seed}..."
        
        sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ns2d_sc_${strat}_${seed}
#SBATCH --account=euhpc_d36_033
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=${run_dir}/%j.out
#SBATCH --error=${run_dir}/%j.err

set -euo pipefail
cd /leonardo/home/userexternal/pcesar00/ogas_states

export OMP_NUM_THREADS=8
export JAX_PLATFORMS=cuda,cpu

module load profile/deeplrn 2>/dev/null || true
module load cuda/12.3 2>/dev/null || module load cuda/12.1 2>/dev/null || true

export PATH="/leonardo/prod/opt/compilers/cuda/12.3/none/bin:\$PATH"
export LD_LIBRARY_PATH="/leonardo/prod/opt/compilers/cuda/12.3/none/lib64:\${LD_LIBRARY_PATH:-}"

source .venv/bin/activate

python -u scripts/run_2d_navier_stokes_scaled.py \
    --strategy "${strat}" \
    --seed ${seed} \
    --rounds 5 \
    --epochs 10 \
    --n_traj 32 \
    --steps 20 \
    --output_dir "${run_dir}"
EOF

        sleep 0.2
    done
done

echo "=== ALL 25 SCALED 2D NAVIER-STOKES JOBS PROCESSED ==="
