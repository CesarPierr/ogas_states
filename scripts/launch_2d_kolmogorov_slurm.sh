#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ns2d_kolmogorov_pilot"
mkdir -p "$BASE_DIR"

SCENARIOS=(
    "uniform_baseline"
    "heuristic_tube"
    "classic_al_topk"
    "classic_al_sbal"
    "ogas_generative"
)

SEED=101

for strat in "${SCENARIOS[@]}"; do
    run_dir="${BASE_DIR}/${strat}_seed${SEED}"
    mkdir -p "$run_dir"
    
    echo "Submitting 2D NS Kolmogorov Pilot: strategy=${strat}, seed=${SEED}..."
    
    sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ns2d_${strat}_${SEED}
#SBATCH --account=EUHPC_D36_033
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=${run_dir}/%j.out
#SBATCH --error=${run_dir}/%j.err

set -euo pipefail
cd /leonardo/home/userexternal/pcesar00/ogas_states

export OMP_NUM_THREADS=8
export JAX_PLATFORMS=cpu

module load profile/deeplrn
module load cuda/12.3

source .venv/bin/activate

python -u scripts/run_2d_kolmogorov_pilot.py \
    --strategy "${strat}" \
    --seed ${SEED} \
    --output_dir "${run_dir}"
EOF

    sleep 0.5
done

echo "=== ALL 5 2D NAVIER-STOKES PILOT JOBS SUBMITTED TO SLURM ==="
