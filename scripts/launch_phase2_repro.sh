#!/usr/bin/env bash
set -euo pipefail

# Launch script for Phase 2 repro on Leonardo
# 6 variants x 5 seeds = 30 jobs

CONFIG="configs/leonardo_ks_phase2_repro.yaml"
WANDB_PROJECT="poolbased-vague2-leonardo"

SEEDS=(101 202 303 404 505)

submit() {
    local variant_name=$1
    shift
    local args=("$@")
    
    for seed in "${SEEDS[@]}"; do
        run_dir="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_phase2_repro/${variant_name}_seed${seed}"
        
        echo "Submitting ${variant_name} (seed ${seed})..."
        export LEONARDO_RUN_DIR="$run_dir"
        export LEONARDO_JOB_NAME="${variant_name}_${seed}"
        
        ./scripts/submit_leonardo.sh "$CONFIG" \
            --set "seed=$seed" \
            --set "output_dir=$run_dir" \
            --set "wandb.project=$WANDB_PROJECT" \
            "${args[@]}"
            
        sleep 1
    done
}

# 1. Uniform Baseline (Control)
submit "uniform_baseline" "--set" "pool.uniform_fraction=1.0" "--set" "ddpm.enabled=false"

# 2. tophalf 0.0
submit "tophalf_0p0" "--set" "pool.uniform_fraction=0.0" "--set" "ddpm.difficulty_signal=loss"

# 3. tophalf 0.25
submit "tophalf_0p25" "--set" "pool.uniform_fraction=0.25" "--set" "ddpm.difficulty_signal=loss"

# 4. tophalf 0.5
submit "tophalf_0p5" "--set" "pool.uniform_fraction=0.5" "--set" "ddpm.difficulty_signal=loss"

# 5. tophalf 0.75
submit "tophalf_0p75" "--set" "pool.uniform_fraction=0.75" "--set" "ddpm.difficulty_signal=loss"

# 6. ensvar 0.5
submit "ensvar_0p5" "--set" "pool.uniform_fraction=0.5" "--set" "ddpm.difficulty_signal=ensemble_var"

echo "All jobs submitted!"
