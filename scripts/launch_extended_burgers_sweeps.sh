#!/usr/bin/env bash
# ==============================================================================
# Master Sweep Launcher: Extended Physics Benchmarks (Forced & KdV-Burgers)
# ==============================================================================
# Benchmarks:
#   1. Forced Burgers (Burgulence) - Stationary Shock Collisions
#   2. KdV-Burgers - Dispersive Soliton Trains + Viscous Shock Waves
#
# Methods (5 Strategies):
#   - uniform_baseline
#   - heuristic_tube
#   - classic_al_topk (Ensemble Disagreement / Uncertainty)
#   - classic_al_sbal (Ensemble Disagreement / Uncertainty)
#   - ogas_generative (Flow Matching with Uncertainty Condition: ensemble_var)
#
# Ensemble Sizes: M in {2, 3, 5}
# Seeds: 10 seeds (101..1010)
# ==============================================================================
set -euo pipefail

ALL_SEEDS=(101 202 303 404 505 606 707 808 909 1010)
BASE_SCRATCH="/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs"

export LEONARDO_WALLTIME="04:00:00"
export LEONARDO_PRECREATE_VALIDATION=0

submit_job() {
    local config_file=$1
    local sweep_name=$2
    local variant_name=$3
    local ensemble_size=$4
    local seeds_str=$5
    shift 5
    local extra_args=("$@")
    
    local base_dir="${BASE_SCRATCH}/${sweep_name}_M${ensemble_size}"
    mkdir -p "$base_dir"
    
    read -r -a seeds_array <<< "$seeds_str"
    for seed in "${seeds_array[@]}"; do
        local run_dir="${base_dir}/${variant_name}_seed${seed}"
        mkdir -p "$run_dir"
        
        if [ -f "$run_dir/history.json" ] && [ -f "$run_dir/checkpoint_latest.pt" ]; then
            echo "  [SKIP - ALREADY DONE] ${sweep_name}_M${ensemble_size}/${variant_name}_seed${seed}"
            continue
        fi
        
        echo "  [SUBMIT] ${sweep_name} M=${ensemble_size} | ${variant_name} (seed ${seed})..."
        export LEONARDO_RUN_DIR="$run_dir"
        export LEONARDO_JOB_NAME="${sweep_name:0:4}_m${ensemble_size}_${variant_name:0:6}_${seed}"
        
        ./scripts/submit_leonardo.sh "$config_file" \
            --fresh \
            --set "seed=$seed" \
            --set "output_dir=$run_dir" \
            --set "surrogate.ensemble_size=$ensemble_size" \
            --set "ddpm.difficulty_signal=ensemble_var" \
            --set "wandb.project=${sweep_name}" \
            --set "wandb.group=${variant_name}_M${ensemble_size}" \
            "${extra_args[@]}"
            
        sleep 0.2
    done
}

launch_benchmark() {
    local config_file=$1
    local sweep_name=$2
    local m_size=$3
    
    echo "----------------------------------------------------------------------------"
    echo ">>> Launching ${sweep_name} (M=${m_size}, Uncertainty Evaluation, 10 Seeds) <<<"
    echo "----------------------------------------------------------------------------"
    
    # 1. Uniform Baseline
    submit_job "$config_file" "$sweep_name" "uniform_baseline" "$m_size" "${ALL_SEEDS[*]}" \
        --set "pool.uniform_fraction=1.0" \
        --set "pool.strategy=generator" \
        --set "ddpm.enabled=false"

    # 2. Heuristic Random Tube Baseline
    submit_job "$config_file" "$sweep_name" "heuristic_tube" "$m_size" "${ALL_SEEDS[*]}" \
        --set "pool.uniform_fraction=0.5" \
        --set "pool.strategy=random_tube" \
        --set "ddpm.enabled=false"

    # 3. Classic AL Top-K (Uncertainty / Disagreement)
    submit_job "$config_file" "$sweep_name" "classic_al_topk" "$m_size" "${ALL_SEEDS[*]}" \
        --set "pool.uniform_fraction=0.5" \
        --set "pool.strategy=classic_al_topk" \
        --set "pool.classic_al_oversample=10" \
        --set "ddpm.enabled=false"

    # 4. Classic AL SBAL (Uncertainty / Disagreement)
    submit_job "$config_file" "$sweep_name" "classic_al_sbal" "$m_size" "${ALL_SEEDS[*]}" \
        --set "pool.uniform_fraction=0.5" \
        --set "pool.strategy=classic_al_sbal" \
        --set "pool.classic_al_oversample=10" \
        --set "pool.sbal_alpha=1.0" \
        --set "ddpm.enabled=false"

    # 5. OGAS Generative (Flow Matching with Uncertainty: ensemble_var)
    submit_job "$config_file" "$sweep_name" "ogas_generative" "$m_size" "${ALL_SEEDS[*]}" \
        --set "pool.uniform_fraction=0.5" \
        --set "pool.strategy=generator" \
        --set "ddpm.enabled=true"
}

# Main Execution
TARGET_BENCHMARKS=("forced_burgers" "kdv_burgers")
ENSEMBLE_SIZES=(2 3 5)

if [ $# -gt 0 ]; then
    ENSEMBLE_SIZES=("$@")
fi

for bench in "${TARGET_BENCHMARKS[@]}"; do
    cfg="configs/1d_burgers/${bench}.yaml"
    for m in "${ENSEMBLE_SIZES[@]}"; do
        launch_benchmark "$cfg" "${bench}_sweep" "$m"
    done
done

echo "============================================================================"
echo "=== All Extended Physics Benchmark Jobs Submitted Successfully! ==="
echo "============================================================================"
