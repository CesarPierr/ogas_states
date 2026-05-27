#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=${POOLBASED_ENV:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv}
EXTERNAL_DIR=${POOLBASED_EXTERNAL:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/external}
PYTHON="$ENV_DIR/bin/python"

mkdir -p "$EXTERNAL_DIR"

clone_or_update() {
  local url=$1
  local dst=$2
  local ref=$3
  if [ ! -d "$dst/.git" ]; then
    git clone "$url" "$dst"
  else
    git -C "$dst" fetch --all --prune
  fi
  git -C "$dst" checkout "$ref"
}

clone_or_update https://github.com/dmusekamp/al4pde.git "$EXTERNAL_DIR/al4pde" c183431e43c122717beaf23cbd7f77aebcaaf5c2
clone_or_update https://github.com/microsoft/pdearena.git "$EXTERNAL_DIR/pdearena" 22360a766387c3995220b4a1265a936ab9a81b88
clone_or_update https://github.com/google/jax-cfd.git "$EXTERNAL_DIR/jax-cfd" d215f13282bd63045fb3455f8fac061653428040

"$PYTHON" -m ensurepip --upgrade
"$PYTHON" -m pip install \
  "jax[cuda12]" \
  tensordict \
  omegaconf \
  tree-math \
  dm-haiku \
  einops \
  gin-config
"$PYTHON" -m pip install --no-deps -e "$EXTERNAL_DIR/jax-cfd" -e "$EXTERNAL_DIR/pdearena" -e "$EXTERNAL_DIR/al4pde"

"$PYTHON" - <<'PY'
from poolbased_surrogate.al4pde_bridge import ensure_al4pde_paths
ensure_al4pde_paths()
from al4pde.tasks.sim.ks_jax import ParametricKSJaxSim
from al4pde.tasks.sim.burgers import BurgersSim
from al4pde.modules.unet_cond_1d import Unet1D
print("AL4PDE 1D dependencies are importable.")
PY
