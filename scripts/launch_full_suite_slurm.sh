#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/leonardo_ks_v3_base.yaml"
ALL_SEEDS=(101 202 303 404 505 606 707 808 909 1010)
HEUR_M1_SEEDS=(202 303 404 505 606 707 808 909 1010)

BASE_HEUR="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_heuristic_tube"
BASE_M1="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_m1_improvements"

export LEONARDO_WALLTIME="08:00:00"
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
        
        # Check if already completed (history.json with 10 rounds)
        if [ -f "$run_dir/history.json" ] && [ -f "$run_dir/checkpoint_latest.pt" ]; then
            echo "  [SKIP - ALREADY DONE] ${variant_name}_seed${seed}"
            continue
        fi
        
        echo "  [SUBMIT] ${variant_name} (seed ${seed})..."
        export LEONARDO_RUN_DIR="$run_dir"
        export LEONARDO_JOB_NAME="${variant_name}_${seed}"
        
        ./scripts/submit_leonardo.sh "$CONFIG" \
            --fresh \
            --set "seed=$seed" \
            --set "output_dir=$run_dir" \
            --set "wandb.project=ks800-al-improvements" \
            --set "wandb.group=$variant_name" \
            "${extra_args[@]}"
        
        sleep 0.5
    done
}

echo "=== 1. SUBMITTING HEURISTIC BASELINES (random_tube) ==="

# Heuristic M=1 (seed 101 is already completed)
submit_arm "$BASE_HEUR" "ens1_heuristic_tube" "${HEUR_M1_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=random_tube" \
    --set "ddpm.enabled=false"

# Heuristic M=3
submit_arm "$BASE_HEUR" "ens3_heuristic_tube" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=random_tube" \
    --set "ddpm.enabled=false"

# Heuristic M=5
submit_arm "$BASE_HEUR" "ens5_heuristic_tube" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=5" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=random_tube" \
    --set "ddpm.enabled=false"

echo "=== 2. SUBMITTING M=1 IMPROVEMENTS ==="

# Sobolev H1 Regularization
submit_arm "$BASE_M1" "ens1_loss_sobolev_tv" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "surrogate.sobolev_weight=0.05" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=scratch"

# Jitter Replay (Gaussian input noise on training)
submit_arm "$BASE_M1" "ens1_loss_jitter_replay" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "surrogate.input_noise_std=0.05" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=scratch"

# Spectral Edit Smoothing (Edit Mode)
submit_arm "$BASE_M1" "ens1_loss_edit_smooth" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=edit"

# Pushforward Multi-Step Regularization
submit_arm "$BASE_M1" "ens1_loss_pushforward" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "surrogate.pushforward_steps=1" \
    --set "surrogate.pushforward_weight=0.5" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=scratch"

echo "=== ALL JOBS SUBMITTED TO SLURM SUCCESSFULLY ==="
