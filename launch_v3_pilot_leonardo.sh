#!/usr/bin/env bash
# V3 pilot on Leonardo — same 6 arms as launch_v3_pilot.sh, SLURM edition.
# Prereqs (see LEONARDO.md): venv installed, validation bank + diverse suite built,
# configs/leonardo_ks_v3_base.yaml present (paths under $FAST).
set -euo pipefail

export LEONARDO_WALLTIME="16:00:00"
export LEONARDO_PRECREATE_VALIDATION=0   # bank must already exist (see LEONARDO.md)

BASE_CFG="configs/leonardo_ks_v3_base.yaml"
BASE_DIR="$FAST/ogas_states_runs/ks800_v3_pilot"
SEED=101

launch_job() {
  local name=$1; shift
  export LEONARDO_RUN_DIR="$BASE_DIR/${name}_seed${SEED}"
  mkdir -p "$LEONARDO_RUN_DIR"
  export LEONARDO_JOB_NAME="v3_${name}_${SEED}"
  echo "Submitting $name seed $SEED"
  ./scripts/submit_leonardo.sh "$BASE_CFG" \
    --fresh \
    --set seed="$SEED" \
    --set output_dir="$LEONARDO_RUN_DIR" \
    --set wandb.group="ks800_v3_${name}" \
    "$@"
}

launch_job uniform_baseline --set pool.uniform_fraction=1.0 --set ddpm.enabled=false
launch_job noise_inject     --set pool.uniform_fraction=1.0 --set ddpm.enabled=false --set surrogate.input_noise_std=0.25
launch_job random_tube      --set pool.uniform_fraction=0.5 --set pool.strategy=random_tube --set ddpm.enabled=false
launch_job mined_ic         --set pool.uniform_fraction=0.5 --set pool.strategy=mined_ic --set ddpm.enabled=false
launch_job gen_v3           --set pool.uniform_fraction=0.5
launch_job gen_v3_edit      --set pool.uniform_fraction=0.5 --set ddpm.sample_mode=edit --set ddpm.edit_t0=0.6

echo "All V3 pilot jobs submitted. Monitor: squeue -u \$USER"
