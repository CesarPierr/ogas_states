#!/usr/bin/env bash
set -euo pipefail
# Phase 3 (docs/roadmap.md) -- multi-PDE generality. Replicates the clean Phase 2 KS sweep
# on 1D Burgers (shocks = clean hard states): (A) uniform_fraction Pareto sweep {0,.25,.5,.75,1},
# (B) difficulty signal loss vs ensemble disagreement at uf=0.5. Surrogate hidden64 + ensemble2,
# rounds=10 hard cap, flow_matching nq20 top_half generator. 5 seeds each.
# uf=1.0 = pure-uniform baseline (generator disabled). Reuses the burgers_res256 validation bank
# + burgers_hard_sub5 hard/tube/cov sets (built by oar-burgers-build-val).

export BIGFOOT_WALLTIME="12:00:00"      # res256 Burgers is cheaper than res800 KS
export BIGFOOT_PRECREATE_VALIDATION=0   # validation npz already exists

BASE_CFG="configs/bigfoot_burgers_phase3_base.yaml"
BASE_DIR="/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/burgers256_phase3_5seed"
SEEDS=(101 202 303 404 505)

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
  export BIGFOOT_JOB_NAME="p3b_${name}_${seed}"
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
echo "All Phase 3 (Burgers) jobs submitted."
