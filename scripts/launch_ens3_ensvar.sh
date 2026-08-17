#!/usr/bin/env bash
set -euo pipefail

SEEDS=(101 202 303 404 505 606 707 808 909 1010)
CONFIG="configs/leonardo_ks_v3_base.yaml"
WANDB_PROJECT="poolbased-ensemble-scaling"
BASE_DIR="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_ensemble_scaling"

export LEONARDO_WALLTIME="12:00:00"
export LEONARDO_PRECREATE_VALIDATION=0

submit() {
    local variant_name=$1
    shift
    local args=("$@")
    
    for seed in "${SEEDS[@]}"; do
        run_dir="${BASE_DIR}/${variant_name}_seed${seed}"
        mkdir -p "$run_dir"
        
        echo "Submitting ${variant_name} (seed ${seed})..."
        export LEONARDO_RUN_DIR="$run_dir"
        export LEONARDO_JOB_NAME="ens_${variant_name}_${seed}"
        
        ./scripts/submit_leonardo.sh "$CONFIG" \
            --fresh \
            --set "seed=$seed" \
            --set "output_dir=$run_dir" \
            --set "wandb.project=$WANDB_PROJECT" \
            --set "wandb.group=$variant_name" \
            "${args[@]}"
            
        sleep 1
    done
}

submit "ens3_ensvar_0p5" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=0.5" \
    --set "ddpm.difficulty_signal=ensemble_var" \
    --set "ddpm.sample_mode=scratch"

echo "ens3_ensvar_0p5 10 seeds submitted successfully!"
