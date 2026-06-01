#!/bin/bash
export BIGFOOT_JOB_NAME="poolbased-bigfoot"
export BIGFOOT_WALLTIME="03:00:00"
export BIGFOOT_PRECREATE_VALIDATION=0
BASE_DIR="/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/ks800_param5case_5seed"

launch_job() {
  local cfg=$1
  local name=$2
  local seed=$3
  local mode_flag=${4:-}
  local mode_val=${5:-}
  
  export BIGFOOT_RUN_DIR="$BASE_DIR/${name}_seed${seed}"
  mkdir -p "$BIGFOOT_RUN_DIR"
  export BIGFOOT_JOB_NAME="${name}_${seed}"
  
  echo "Submitting $name seed $seed"
  if [ -n "$mode_flag" ]; then
    ./scripts/submit_bigfoot.sh "$cfg" --set seed="$seed" --set output_dir="$BIGFOOT_RUN_DIR" --set wandb.group="ks800_5case_${name}" "$mode_flag" "$mode_val"
  else
    ./scripts/submit_bigfoot.sh "$cfg" --set seed="$seed" --set output_dir="$BIGFOOT_RUN_DIR" --set wandb.group="ks800_5case_${name}"
  fi
}

for seed in 101 202 303 404 505; do
  launch_job "configs/bigfoot_ks_uniform.yaml" "uniform" "$seed"
  launch_job "configs/bigfoot_ks_mixed.yaml" "loss_paramcond" "$seed" "--set" "ddpm.param_mode=condition"
  launch_job "configs/bigfoot_ks_mixed.yaml" "loss_paramgen" "$seed" "--set" "ddpm.param_mode=generate"
  launch_job "configs/bigfoot_ks_density.yaml" "density_paramcond" "$seed" "--set" "ddpm.param_mode=condition"
  launch_job "configs/bigfoot_ks_density.yaml" "density_paramgen" "$seed" "--set" "ddpm.param_mode=generate"
done
