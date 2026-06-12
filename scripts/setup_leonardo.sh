#!/usr/bin/env bash
# One-shot environment setup for Leonardo (CINECA). Run ON the Leonardo login node,
# from the repo root ($FAST/ogas_states). Safe to re-run (idempotent).
#
#   cd $FAST/ogas_states && bash scripts/setup_leonardo.sh
#
# Does: venv + deps, external AL4PDE checkouts, Leonardo config generation,
# validation bank + hard/tube/diverse suites (CPU, ~1-2h total).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
: "${FAST:?FAST is not set — run on Leonardo (module env provides \$FAST)}"

ENV_DIR="$ROOT/.venv"
EXTERNAL_DIR="$FAST/ogas_external"
VAL_DIR="$FAST/ogas_validation"
mkdir -p "$EXTERNAL_DIR" "$VAL_DIR" "$FAST/ogas_states_runs"

echo "== 1/5 venv =="
module purge
module load python/3.11 2>/dev/null || module load python 2>/dev/null || true
if [ ! -x "$ENV_DIR/bin/python" ]; then
  python3.11 -m venv "$ENV_DIR" 2>/dev/null || python3 -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install -q --upgrade pip
"$ENV_DIR/bin/python" -m pip install -q -e .

# Make the bridge find the external checkouts (default path is the Bigfoot one).
# Must be exported BEFORE step 2: the installer's import check relies on them
# (the pdearena editable install is a no-op finder; sys.path needs the repo roots).
export AL4PDE_ROOT="$EXTERNAL_DIR/al4pde"
export PDEARENA_ROOT="$EXTERNAL_DIR/pdearena"
export JAX_CFD_ROOT="$EXTERNAL_DIR/jax-cfd"
grep -q AL4PDE_ROOT "$ENV_DIR/bin/activate" || cat >> "$ENV_DIR/bin/activate" <<EOF
export AL4PDE_ROOT="$EXTERNAL_DIR/al4pde"
export PDEARENA_ROOT="$EXTERNAL_DIR/pdearena"
export JAX_CFD_ROOT="$EXTERNAL_DIR/jax-cfd"
EOF

echo "== 2/5 external deps (al4pde/pdearena/jax-cfd + pinned torch/jax) =="
POOLBASED_ENV="$ENV_DIR" POOLBASED_EXTERNAL="$EXTERNAL_DIR" bash scripts/install_al4pde_deps.sh

echo "== 3/5 leonardo config from v3 base =="
# Rewrite the /bettik paths of the Bigfoot config to the Leonardo layout.
sed -e "s|/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_validation|$VAL_DIR|g" \
    -e "s|/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs|$FAST/ogas_states_runs|g" \
    configs/bigfoot_ks_v3_base.yaml > configs/leonardo_ks_v3_base.yaml
echo "wrote configs/leonardo_ks_v3_base.yaml"

echo "== 4/5 validation bank (KS sub5, seeded — identical to Bigfoot's) =="
BANK="$VAL_DIR/ks_res800_al4pde_sub5_seed43_n1500_t100.npz"
if [ ! -f "$BANK" ]; then
  JAX_PLATFORMS=cpu "$ENV_DIR/bin/python" -u scripts/generate_validation_sub5.py \
    --output "$BANK" --seed 43 --n-trajectories 1500 --trajectory-steps 100 \
    --n-substeps 5 --n-warmup 20 --param1-min 10.0 --batch-size 50
else
  echo "bank exists, skipping"
fi

echo "== 5/5 diverse hard/tube suite =="
if [ ! -d "$VAL_DIR/ks_diverse_suite" ] || [ -z "$(ls -A "$VAL_DIR/ks_diverse_suite" 2>/dev/null)" ]; then
  JAX_PLATFORMS=cpu "$ENV_DIR/bin/python" -u scripts/build_diverse_validation.py \
    --bank "$BANK" --output-dir "$VAL_DIR/ks_diverse_suite" \
    --config configs/leonardo_ks_v3_base.yaml --seed 0
else
  echo "suite exists, skipping"
fi

echo "== smoke test (CPU, ~5 min) =="
JAX_PLATFORMS=cpu "$ENV_DIR/bin/python" -u -m poolbased_surrogate.run configs/smoke_v3.yaml \
  --fresh --set output_dir=/tmp/smoke_v3_leo --set pool.rounds=1
echo "SETUP COMPLETE. Launch the pilot with: bash launch_v3_pilot_leonardo.sh"
