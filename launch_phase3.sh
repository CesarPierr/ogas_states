#!/usr/bin/env bash
set -euo pipefail
# Phase 3 (docs/roadmap.md) -- Burgers 256 5-seed sweep.
# Evaluates uniform_fraction sweep {0, .25, .5, .75, 1} and the ensemble disagreement
# signal (ensemble_var) at uf=0.5. Matches Phase 2 KS hyperparameters.

export BIGFOOT_WALLTIME="16:00:00"      # Safe walltime
export BIGFOOT_PRECREATE_VALIDATION=0   # validation npz already exists

BASE_CFG="configs/bigfoot_burgers_phase3_base.yaml"
BASE_DIR="/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/burgers256_phase3_5seed"
SEEDS=(101 202 303 404 505)

# name -> "extra --set args" (uniform_fraction sweep + difficulty signal axis)
declare -A VARIANTS=(
  [uf00_tophalf]="--set pool.uniform_fraction=0.0"
  [uf25_tophalf]="--set pool.uniform_fraction=0.25"
  [uf50_tophalf]="--set pool.uniform_fraction=0.5"
  [uf75_tophalf]="--set pool.uniform_fraction=0.75"
  [uniform_baseline]="--set pool.uniform_fraction=1.0 --set ddpm.enabled=false"
  [uf50_ensvar]="--set pool.uniform_fraction=0.5 --set ddpm.difficulty_signal=ensemble_var"
)

launch_job() {
  local name=$1 seed=$2; shift 2
  export BIGFOOT_RUN_DIR="$BASE_DIR/${name}_seed${seed}"
  mkdir -p "$BIGFOOT_RUN_DIR"
  export BIGFOOT_JOB_NAME="p3_${name}_${seed}"
  echo "Submitting $name seed $seed"
  ./scripts/submit_bigfoot.sh "$BASE_CFG" \
    --fresh \
    --set seed="$seed" \
    --set output_dir="$BIGFOOT_RUN_DIR" \
    "$@"
}

for seed in "${SEEDS[@]}"; do
  for name in "${!VARIANTS[@]}"; do
    # shellcheck disable=SC2086
    launch_job "$name" "$seed" ${VARIANTS[$name]}
  done
done
echo "All Phase 3 Burgers jobs submitted."
