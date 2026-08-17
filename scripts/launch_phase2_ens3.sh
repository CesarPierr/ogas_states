#!/usr/bin/env bash
set -euo pipefail

# Launch script for Phase 2 ens3 on Leonardo
# 2 variants x 5 seeds = 10 jobs

CONFIG="configs/leonardo_ks_phase2_repro.yaml"
WANDB_PROJECT="poolbased-vague2-leonardo-ens3"

SEEDS=(101 202 303 404 505)

submit() {
    local variant_name=$1
    shift
    local args=("$@")
    
    for seed in "${SEEDS[@]}"; do
        run_dir="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_phase2_ens3/${variant_name}_seed${seed}"
        
        echo "Submitting ${variant_name} (seed ${seed})..."
        export LEONARDO_RUN_DIR="$run_dir"
        export LEONARDO_JOB_NAME="${variant_name}_${seed}_ens3"
        
        ./scripts/submit_leonardo.sh "$CONFIG" \
            --set "seed=$seed" \
            --set "output_dir=$run_dir" \
            --set "wandb.project=$WANDB_PROJECT" \
            --set "surrogate.ensemble_size=3" \
            "${args[@]}"
            
        sleep 1
    done
}

# 1. Uniform Baseline (Control)
submit "uniform_baseline" "--set" "pool.uniform_fraction=1.0" "--set" "ddpm.enabled=false"

# 2. ensvar 0.5
submit "ensvar_0p5" "--set" "pool.uniform_fraction=0.5" "--set" "ddpm.difficulty_signal=ensemble_var"

echo "All jobs submitted!"
