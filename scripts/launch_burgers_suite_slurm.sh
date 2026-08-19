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

echo "=========================================================="
echo "=== LAUNCHING COMPLETE BURGERS 1D BENCHMARK (10 SEEDS) ==="
echo "=========================================================="

echo "=== 1. M=1 SINGLE MODELS ==="

# 1.1 Uniform baseline control
submit_arm "$BASE_BURGERS" "ens1_uniform_baseline" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=1.0" \
    --set "pool.strategy=generator" \
    --set "ddpm.enabled=false"

# 1.2 Heuristic random tube baseline
submit_arm "$BASE_BURGERS" "ens1_heuristic_tube" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=random_tube" \
    --set "ddpm.enabled=false"

# 1.3 Active Learning (scratch loss-spectrum)
submit_arm "$BASE_BURGERS" "pure_scratch_loss_spectrum" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=scratch"

# 1.4 Jitter Replay
submit_arm "$BASE_BURGERS" "ens1_loss_jitter_replay" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "surrogate.input_noise_std=0.05" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=scratch"

# 1.5 Sobolev H1 TV Regularization
submit_arm "$BASE_BURGERS" "ens1_loss_sobolev_tv" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "surrogate.sobolev_weight=0.05" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=scratch"

# 1.6 Pushforward Multi-Step Regularization
submit_arm "$BASE_BURGERS" "ens1_loss_pushforward" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=1" \
    --set "surrogate.pushforward_steps=1" \
    --set "surrogate.pushforward_weight=0.5" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=scratch"

echo "=== 2. M=3 ENSEMBLES ==="

# 2.1 Uniform baseline control
submit_arm "$BASE_BURGERS" "ens3_uniform_baseline" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=1.0" \
    --set "pool.strategy=generator" \
    --set "ddpm.enabled=false"

# 2.2 Heuristic random tube baseline
submit_arm "$BASE_BURGERS" "ens3_heuristic_tube" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=random_tube" \
    --set "ddpm.enabled=false"

# 2.3 Ensemble Variance Active Learning
submit_arm "$BASE_BURGERS" "ens3_ensvar_0p5" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=ensemble_var" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=scratch"

# 2.4 Gen-V3 Edit Mode Active Learning
submit_arm "$BASE_BURGERS" "ens3_gen_v3_edit" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=ensemble_var" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=edit"

echo "=== 3. M=5 ENSEMBLES ==="

# 3.1 Uniform baseline control
submit_arm "$BASE_BURGERS" "ens5_uniform_baseline" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=5" \
    --set "pool.uniform_fraction=1.0" \
    --set "pool.strategy=generator" \
    --set "ddpm.enabled=false"

# 3.2 Heuristic random tube baseline
submit_arm "$BASE_BURGERS" "ens5_heuristic_tube" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=5" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=random_tube" \
    --set "ddpm.enabled=false"

# 3.3 Ensemble Variance Active Learning
submit_arm "$BASE_BURGERS" "ens5_ensvar_0p5" "${ALL_SEEDS[*]}" \
    --set "surrogate.ensemble_size=5" \
    --set "pool.uniform_fraction=0.5" \
    --set "pool.strategy=generator" \
    --set "ddpm.difficulty_signal=ensemble_var" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.sample_mode=scratch"

echo "=========================================================="
echo "=== ALL BURGERS EXPERIMENTS SUBMITTED SUCCESSFULLY ==="
echo "=========================================================="
