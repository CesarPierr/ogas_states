#!/usr/bin/env bash
set -euo pipefail

# Launch script for Ensemble Scaling study on Leonardo (M=3 and M=5)
# 4 variants x 5 seeds = 20 jobs

CONFIG="configs/leonardo_ks_v3_base.yaml"
WANDB_PROJECT="poolbased-ensemble-scaling"
BASE_DIR="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_ensemble_scaling"

if [ "$#" -eq 0 ]; then
  SEEDS=(101 202 303 404 505)
else
  SEEDS=("$@")
fi

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

# --- Ensemble M=3 complements ---
# 1. Uniform Baseline with M=3
submit "ens3_uniform_baseline" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=1.0" \
    --set "ddpm.enabled=false"

# 2. gen_v3_edit with M=3
submit "ens3_gen_v3_edit" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=0.5" \
    --set "ddpm.sample_mode=edit" \
    --set "ddpm.edit_t0=0.6"

# --- Ensemble M=5 full sweep ---
# 2. Uniform Baseline with M=5
submit "ens5_uniform_baseline" \
    --set "surrogate.ensemble_size=5" \
    --set "pool.uniform_fraction=1.0" \
    --set "ddpm.enabled=false"

# 3. ensvar_0p5 (V2 pure disagreement active learning) with M=5
submit "ens5_ensvar_0p5" \
    --set "surrogate.ensemble_size=5" \
    --set "pool.uniform_fraction=0.5" \
    --set "ddpm.difficulty_signal=ensemble_var" \
    --set "ddpm.sample_mode=scratch"

# 4. gen_v3_edit (V3 SDEdit + candidate TV filter) with M=5
submit "ens5_gen_v3_edit" \
    --set "surrogate.ensemble_size=5" \
    --set "pool.uniform_fraction=0.5" \
    --set "ddpm.sample_mode=edit" \
    --set "ddpm.edit_t0=0.6"

echo "All ensemble scaling jobs submitted successfully!"
