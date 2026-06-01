#!/usr/bin/env bash
set -euo pipefail
# Experiment v3: balance the RMSE/NRMSE tradeoff. Best arch (fm_s64_h64 + nq20), 30 rounds
# x 10 epochs. Two axes: (1) uniform_fraction (bulk coverage to recover RMSE),
# (2) NRMSE conditioning (rank difficulty by relative error). 5 seeds each.

export BIGFOOT_WALLTIME="12:00:00"
export BIGFOOT_PRECREATE_VALIDATION=0

BASE_DIR="/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/ks800_sub5_uf_nrmse_5seed"
SEEDS=(101 202 303 404 505)

declare -A VARIANTS=(
  [uf10_tophalf]="configs/bigfoot_ks_sub5_uf10_tophalf.yaml"
  [uf50_tophalf]="configs/bigfoot_ks_sub5_uf50_tophalf.yaml"
  [nrmse_tophalf]="configs/bigfoot_ks_sub5_nrmse_tophalf.yaml"
  [nrmse_uf50_tophalf]="configs/bigfoot_ks_sub5_nrmse_uf50_tophalf.yaml"
  [nrmse_expbias_t03]="configs/bigfoot_ks_sub5_nrmse_expbias_t03.yaml"
  [uf50_expbias_t03]="configs/bigfoot_ks_sub5_uf50_expbias_t03.yaml"
)

launch_job() {
  local name=$1 cfg=$2 seed=$3
  export BIGFOOT_RUN_DIR="$BASE_DIR/${name}_seed${seed}"
  mkdir -p "$BIGFOOT_RUN_DIR"
  export BIGFOOT_JOB_NAME="ufn_${name}_${seed}"
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
