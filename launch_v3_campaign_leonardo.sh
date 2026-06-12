#!/usr/bin/env bash
# V3 5-seed campaign on Leonardo: same 6 arms as the pilot, seeds 202/303/404/505
# (seed 101 = the pilot itself, same run-dir layout, so analysis globs all 5 seeds).
# Usage: bash launch_v3_campaign_leonardo.sh [seed ...]   (default: 202 303 404 505)
set -euo pipefail

export LEONARDO_WALLTIME="${LEONARDO_WALLTIME:-10:00:00}"
export LEONARDO_PRECREATE_VALIDATION=0   # bank must already exist (see LEONARDO.md)

BASE_CFG="configs/leonardo_ks_v3_base.yaml"
BASE_DIR="$FAST/ogas_states_runs/ks800_v3_pilot"
SEEDS=("${@:-202 303 404 505}")
[ $# -eq 0 ] && SEEDS=(202 303 404 505)

launch_job() {
  local name=$1 seed=$2; shift 2
  export LEONARDO_RUN_DIR="$BASE_DIR/${name}_seed${seed}"
  mkdir -p "$LEONARDO_RUN_DIR"
  export LEONARDO_JOB_NAME="v3_${name}_${seed}"
  echo "Submitting $name seed $seed"
  ./scripts/submit_leonardo.sh "$BASE_CFG" \
    --fresh \
    --set seed="$seed" \
    --set output_dir="$LEONARDO_RUN_DIR" \
    --set wandb.group="ks800_v3_${name}" \
    "$@"
}

for SEED in "${SEEDS[@]}"; do
  launch_job uniform_baseline "$SEED" --set pool.uniform_fraction=1.0 --set ddpm.enabled=false
  launch_job noise_inject     "$SEED" --set pool.uniform_fraction=1.0 --set ddpm.enabled=false --set surrogate.input_noise_std=0.25
  launch_job random_tube      "$SEED" --set pool.uniform_fraction=0.5 --set pool.strategy=random_tube --set ddpm.enabled=false
  launch_job mined_ic         "$SEED" --set pool.uniform_fraction=0.5 --set pool.strategy=mined_ic --set ddpm.enabled=false
  launch_job gen_v3           "$SEED" --set pool.uniform_fraction=0.5
  launch_job gen_v3_edit      "$SEED" --set pool.uniform_fraction=0.5 --set ddpm.sample_mode=edit --set ddpm.edit_t0=0.6
done

echo "Campaign submitted. Monitor: squeue -u \$USER"
