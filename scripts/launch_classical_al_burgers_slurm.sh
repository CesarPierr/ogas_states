#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/leonardo_burgers_base.yaml"
ALL_SEEDS=(101 202 303 404 505 606 707 808 909 1010)

BASE_BURGERS="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/burgers_benchmark"

export LEONARDO_WALLTIME="12:00:00"
export LEONARDO_PRECREATE_VALIDATION=0

submit_arm() {
    local base_dir=$1
    local variant_name=$2
    local seeds_str=$3
    shift 3
    local extra_args=("$@")
    
    read -r -a seeds_array <<< "$seeds_str"
    
    for seed in "${seeds_array[@]}"; do
        local run_dir="${base_dir}/${variant_name}_seed${seed}"
        mkdir -p "$run_dir"
        
        # Check if already completed (history.json with 5 rounds)
        if [ -f "$run_dir/history.json" ] && [ -f "$run_dir/checkpoint_latest.pt" ]; then
            echo "  [SKIP - ALREADY DONE] ${variant_name}_seed${seed}"
            continue
        fi
        
        echo "  [SUBMIT] ${variant_name} (seed ${seed})..."
        export LEONARDO_RUN_DIR="$run_dir"
        export LEONARDO_JOB_NAME="burgers_${variant_name}_${seed}"
        
        ./scripts/submit_leonardo.sh "$CONFIG" \
            --fresh \
            --set "seed=$seed" \
            --set "output_dir=$run_dir" \
            --set "wandb.project=burgers-benchmark-generality" \
            --set "wandb.group=$variant_name" \
            "${extra_args[@]}"
        
        sleep 0.5
    done
}

echo "=================================================================="
echo "=== LAUNCHING CLASSICAL ACTIVE LEARNING ON BURGERS 1D (10 SEEDS) ==="
echo "=================================================================="

echo "=== 1. M=1 SINGLE MODEL (Forward Loss Scoring) ==="

# 1.1 M=1 Classic AL Top-K
submit_arm "$BASE_BURGERS" "ens1_classic_al_topk" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=classic_al_topk" \
    --set "pool.classic_al_oversample=10" \
    --set "ddpm.enabled=false"

# 1.2 M=1 Classic AL SBAL (alpha=1)
submit_arm "$BASE_BURGERS" "ens1_classic_al_sbal" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=classic_al_sbal" \
    --set "pool.classic_al_oversample=10" \
    --set "pool.sbal_alpha=1.0" \
    --set "ddpm.enabled=false"

echo "=== 2. M=3 ENSEMBLE (Variance Scoring) ==="

# 2.1 M=3 Classic AL Top-K
submit_arm "$BASE_BURGERS" "ens3_classic_al_topk" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=classic_al_topk" \
    --set "pool.classic_al_oversample=10" \
    --set "ddpm.enabled=false"

# 2.2 M=3 Classic AL SBAL (alpha=1)
submit_arm "$BASE_BURGERS" "ens3_classic_al_sbal" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=classic_al_sbal" \
    --set "pool.classic_al_oversample=10" \
    --set "pool.sbal_alpha=1.0" \
    --set "ddpm.enabled=false"

echo "=== 3. M=5 ENSEMBLE (Variance Scoring) ==="

# 3.1 M=5 Classic AL Top-K
submit_arm "$BASE_BURGERS" "ens5_classic_al_topk" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=5" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=classic_al_topk" \
    --set "pool.classic_al_oversample=10" \
    --set "ddpm.enabled=false"

# 3.2 M=5 Classic AL SBAL (alpha=1)
submit_arm "$BASE_BURGERS" "ens5_classic_al_sbal" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=5" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=classic_al_sbal" \
    --set "pool.classic_al_oversample=10" \
    --set "pool.sbal_alpha=1.0" \
    --set "ddpm.enabled=false"

echo "=================================================================="
echo "=== ALL CLASSICAL AL BURGERS EXPERIMENTS SUBMITTED SUCCESSFULLY ==="
echo "=================================================================="
