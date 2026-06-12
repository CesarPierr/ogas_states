#!/usr/bin/env bash
set -euo pipefail
# V3 pilot -- 6 arms x 1 seed (101), all from configs/bigfoot_ks_v3_base.yaml.
#
# Arms:
#   uniform_baseline  pure uniform pool, no generator, no noise injection
#   noise_inject      uniform pool + Gaussian input-noise injection (0.25 sigma)
#   random_tube       half-pool from attractor anchors + random low-k perturbations
#   mined_ic          half-pool from disagreement-selected random ICs (no generation)
#   gen_v3            full v3 generator (scratch mode, all v3 defaults)
#   gen_v3_edit       v3 generator in SDEdit mode (edit_t0=0.6)
#
# NOTE: arms random_tube and mined_ic pass ddpm.enabled=false with
# pool.uniform_fraction=0.5.  run.py currently raises when ddpm.enabled=false
# and uniform_fraction<1.0 (the check assumes the non-uniform half requires the
# generator).  That check is being relaxed to allow strategy!=generator; these
# arms are correctly specified and will work once the relaxation lands.

export BIGFOOT_WALLTIME="16:00:00"
export BIGFOOT_PRECREATE_VALIDATION=0   # validation npz already exists

BASE_CFG="configs/bigfoot_ks_v3_base.yaml"
BASE_DIR="/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/ks800_v3_pilot"
SEED=101

launch_job() {
  local name=$1; shift
  export BIGFOOT_RUN_DIR="$BASE_DIR/${name}_seed${SEED}"
  mkdir -p "$BIGFOOT_RUN_DIR"
  export BIGFOOT_JOB_NAME="v3_${name}_${SEED}"
  echo "Submitting $name seed $SEED"
  ./scripts/submit_bigfoot.sh "$BASE_CFG" \
    --fresh \
    --set seed="$SEED" \
    --set output_dir="$BIGFOOT_RUN_DIR" \
    --set wandb.group="ks800_v3_${name}" \
    "$@"
}

# (a) pure-uniform baseline
launch_job uniform_baseline \
  --set pool.uniform_fraction=1.0 \
  --set ddpm.enabled=false

# (b) noise-injection baseline (uniform pool + input noise, no generator)
launch_job noise_inject \
  --set pool.uniform_fraction=1.0 \
  --set ddpm.enabled=false \
  --set surrogate.input_noise_std=0.25

# (c) random-tube baseline (non-uniform ICs from attractor perturbations, no generator)
# NOTE: ddpm.enabled=false with uniform_fraction<1.0 -- requires run.py relaxation (see above)
launch_job random_tube \
  --set pool.uniform_fraction=0.5 \
  --set pool.strategy=random_tube \
  --set ddpm.enabled=false

# (d) mined-IC baseline (disagreement-selected random ICs, no generation)
# NOTE: ddpm.enabled=false with uniform_fraction<1.0 -- requires run.py relaxation (see above)
launch_job mined_ic \
  --set pool.uniform_fraction=0.5 \
  --set pool.strategy=mined_ic \
  --set ddpm.enabled=false

# (e) full v3 generator in scratch mode (all v3 defaults from base config)
launch_job gen_v3 \
  --set pool.uniform_fraction=0.5

# (f) v3 generator in SDEdit mode
launch_job gen_v3_edit \
  --set pool.uniform_fraction=0.5 \
  --set ddpm.sample_mode=edit \
  --set ddpm.edit_t0=0.6

echo "All V3 pilot jobs submitted."
