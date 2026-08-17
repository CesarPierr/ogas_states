#!/usr/bin/env bash
set -euo pipefail

# Launch script for the Pure Scratch & One-Step Loss Minimalist Campaign
# Direct sampling from noise without SDEdit, without uncertainty re-filtering
# 4 variants x 5 seeds = 20 jobs

CONFIG="configs/leonardo_ks_v3_base.yaml"
WANDB_PROJECT="poolbased-pure-scratch-loss"
BASE_DIR="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_pure_scratch"

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
        export LEONARDO_JOB_NAME="pure_${variant_name}_${seed}"
        
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

# 1. Single-model Uniform Baseline Control (M=1)
submit "ens1_uniform_baseline" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=1.0" \
    --set "ddpm.enabled=false"

# 2. Pure One-Step Loss from Pure White Noise (M=1, scratch, candidate_factor=1)
submit "pure_scratch_loss_white" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=0.5" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.loss_metric=rmse" \
    --set "ddpm.sample_mode=scratch" \
    --set "ddpm.noise_prior=white" \
    --set "ddpm.candidate_factor=1" \
    --set "ddpm.realism_tv_gate=3.0"

# 3. Pure One-Step Loss from Spectrum Noise Prior (M=1, scratch, candidate_factor=1)
submit "pure_scratch_loss_spectrum" \
    --set "surrogate.ensemble_size=1" \
    --set "pool.uniform_fraction=0.5" \
    --set "ddpm.difficulty_signal=loss" \
    --set "ddpm.loss_metric=rmse" \
    --set "ddpm.sample_mode=scratch" \
    --set "ddpm.noise_prior=spectrum" \
    --set "ddpm.candidate_factor=1" \
    --set "ddpm.realism_tv_gate=3.0"

# 4. Pure Ensemble Disagreement from White Noise (M=3, scratch, candidate_factor=1)
submit "pure_scratch_ensvar_white" \
    --set "surrogate.ensemble_size=3" \
    --set "pool.uniform_fraction=0.5" \
    --set "ddpm.difficulty_signal=ensemble_var" \
    --set "ddpm.sample_mode=scratch" \
    --set "ddpm.noise_prior=white" \
    --set "ddpm.candidate_factor=1" \
    --set "ddpm.realism_tv_gate=3.0"

echo "All pure scratch jobs submitted successfully!"
