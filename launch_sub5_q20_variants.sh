#!/usr/bin/env bash
set -euo pipefail
# Production AL experiment: best flow generator (fm_s64_h64 + nq20, conditional_quantile,
# param_mode=condition, uniform param prior) across sampling laws x train_epochs,
# vs a fully-uniform trajectory baseline. 5 seeds each.

export BIGFOOT_WALLTIME="08:00:00"
export BIGFOOT_PRECREATE_VALIDATION=0   # validation npz already exists

BASE_DIR="/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/ks800_sub5_q20_5seed"
SEEDS=(101 202 303 404 505)

# variant_name -> config file
declare -A VARIANTS=(
  [q20_expbias_e20]="configs/bigfoot_ks_sub5_q20_expbias_e20.yaml"
  [q20_top3_e20]="configs/bigfoot_ks_sub5_q20_top3_e20.yaml"
  [q20_tophalf_e20]="configs/bigfoot_ks_sub5_q20_tophalf_e20.yaml"
  [q20_expbias_e10]="configs/bigfoot_ks_sub5_q20_expbias_e10.yaml"
  [q20_expbias_e30]="configs/bigfoot_ks_sub5_q20_expbias_e30.yaml"
  [uniform]="configs/bigfoot_ks_sub5_uniform.yaml"
)

launch_job() {
  local name=$1 cfg=$2 seed=$3
  export BIGFOOT_RUN_DIR="$BASE_DIR/${name}_seed${seed}"
  mkdir -p "$BIGFOOT_RUN_DIR"
  export BIGFOOT_JOB_NAME="q20_${name}_${seed}"
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
