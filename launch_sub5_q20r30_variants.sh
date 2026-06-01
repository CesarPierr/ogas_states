#!/usr/bin/env bash
set -euo pipefail
# Production AL experiment v2: best flow generator (fm_s64_h64 + nq20, conditional_quantile,
# param_mode=condition, uniform param prior), 30 rounds x 10 epochs/round for finer AL
# dynamics. Sampling-law axis (aggressiveness) vs fully-uniform baseline. 5 seeds each.

export BIGFOOT_WALLTIME="10:00:00"
export BIGFOOT_PRECREATE_VALIDATION=0   # validation npz already exists

BASE_DIR="/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/ks800_sub5_q20r30_5seed"
SEEDS=(101 202 303 404 505)

declare -A VARIANTS=(
  [expbias]="configs/bigfoot_ks_sub5_q20r30_expbias.yaml"
  [expbias_t03]="configs/bigfoot_ks_sub5_q20r30_expbias_t03.yaml"
  [tophalf]="configs/bigfoot_ks_sub5_q20r30_tophalf.yaml"
  [top3]="configs/bigfoot_ks_sub5_q20r30_top3.yaml"
  [uniform]="configs/bigfoot_ks_sub5_q20r30_uniform.yaml"
)

launch_job() {
  local name=$1 cfg=$2 seed=$3
  export BIGFOOT_RUN_DIR="$BASE_DIR/${name}_seed${seed}"
  mkdir -p "$BIGFOOT_RUN_DIR"
  export BIGFOOT_JOB_NAME="r30_${name}_${seed}"
  echo "Submitting $name seed $seed"
  ./scripts/submit_bigfoot.sh "$cfg" \
    --fresh \
    --set seed="$seed" \
    --set output_dir="$BIGFOOT_RUN_DIR"
}

for seed in "${SEEDS[@]}"; do
  for name in "${!VARIANTS[@]}"; do
    launch_job "$name" "${VARIANTS[$name]}" "$seed"
  done
done
echo "All jobs submitted."
