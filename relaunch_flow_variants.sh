#!/usr/bin/env bash
set -euo pipefail

export BIGFOOT_JOB_NAME="poolbased-flow"
export BIGFOOT_WALLTIME="08:00:00"
export BIGFOOT_PRECREATE_VALIDATION=0

BASE_DIR="/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/ks800_flow_variants_clean_5seed"
SEEDS=(101 202 303 404 505)

launch_job() {
  local cfg=$1
  local name=$2
  local seed=$3

  export BIGFOOT_RUN_DIR="$BASE_DIR/${name}_seed${seed}"
  mkdir -p "$BIGFOOT_RUN_DIR"
  export BIGFOOT_JOB_NAME="flow_${name}_${seed}"

  echo "Submitting flow variant $name seed $seed"
  ./scripts/submit_bigfoot.sh "$cfg" \
    --fresh \
    --set seed="$seed" \
    --set output_dir="$BIGFOOT_RUN_DIR" \
    --set wandb.group="ks800_flow_${name}"
}

for seed in "${SEEDS[@]}"; do
  launch_job "configs/bigfoot_ks_flow_tail_paramcond.yaml" "tail_paramcond" "$seed"
  launch_job "configs/bigfoot_ks_flow_high_corner_paramcond.yaml" "high_corner_paramcond" "$seed"
  launch_job "configs/bigfoot_ks_flow_tail_paramgen.yaml" "tail_paramgen" "$seed"
  launch_job "configs/bigfoot_ks_flow_density_paramcond.yaml" "density_paramcond" "$seed"
  launch_job "configs/bigfoot_ks_flow_density_paramgen.yaml" "density_paramgen" "$seed"
done
