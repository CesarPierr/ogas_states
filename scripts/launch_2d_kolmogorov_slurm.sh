#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ns2d_kolmogorov_sweep"
mkdir -p "$BASE_DIR"

SCENARIOS=(
    "uniform_baseline"
    "heuristic_tube"
    "classic_al_topk"
    "classic_al_sbal"
    "ogas_generative"
)

SEEDS=(101 202 303 404 505)

for seed in "${SEEDS[@]}"; do
    for strat in "${SCENARIOS[@]}"; do
        run_dir="${BASE_DIR}/${strat}_seed${seed}"
        mkdir -p "$run_dir"
        
        echo "Submitting 2D NS Kolmogorov: strategy=${strat}, seed=${seed}..."
        
        sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ns2d_${strat}_${seed}
#SBATCH --account=EUHPC_D36_033
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

module load profile/deeplrn
module load cuda/12.3

export PATH="/leonardo/prod/opt/compilers/cuda/12.3/none/bin:$PATH"
export LD_LIBRARY_PATH="/leonardo/prod/opt/compilers/cuda/12.3/none/lib64:${LD_LIBRARY_PATH:-}"

source .venv/bin/activate

python -u scripts/run_2d_kolmogorov_pilot.py \
    --strategy "${strat}" \
    --seed ${seed} \
    --output_dir "${run_dir}"
EOF

        sleep 0.2
    done
done

echo "=== ALL 2D NAVIER-STOKES SWEEP JOBS SUBMITTED TO SLURM ==="

