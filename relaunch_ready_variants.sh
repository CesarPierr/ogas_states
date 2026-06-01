#!/usr/bin/env bash
set -euo pipefail

export BIGFOOT_JOB_NAME="poolbased-ready"
export BIGFOOT_WALLTIME="12:00:00"
export BIGFOOT_PRECREATE_VALIDATION=0

BASE_DIR="/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/ks800_ready_variants_5seed"
SEEDS=(101 202 303 404 505)

launch_job() {
  local cfg=$1
  local name=$2
  local seed=$3

  export BIGFOOT_RUN_DIR="$BASE_DIR/${name}_seed${seed}"
  mkdir -p "$BIGFOOT_RUN_DIR"
  export BIGFOOT_JOB_NAME="ready_${name}_${seed}"

  echo "Submitting ready variant $name seed $seed"
  ./scripts/submit_bigfoot.sh "$cfg" \
    --fresh \
    --set seed="$seed" \
    --set output_dir="$BIGFOOT_RUN_DIR" \
    --set wandb.group="ks800_ready_${name}"
}

for seed in "${SEEDS[@]}"; do
  launch_job "configs/bigfoot_ks_ready_tail_paramcond_res.yaml" "tail_paramcond_res" "$seed"
  launch_job "configs/bigfoot_ks_ready_high_corner_paramcond_res.yaml" "high_corner_paramcond_res" "$seed"
  launch_job "configs/bigfoot_ks_ready_tail_paramgen_res.yaml" "tail_paramgen_res" "$seed"
  launch_job "configs/bigfoot_ks_ready_density_power_paramgen.yaml" "density_power_paramgen" "$seed"
  launch_job "configs/bigfoot_ks_ready_density_power_paramcond_edge.yaml" "density_power_paramcond_edge" "$seed"
done
