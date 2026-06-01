#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${POOLBASED_ENV:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv}
PYTHON_VERSION=${POOLBASED_PYTHON:-3.11}

export UV_LINK_MODE=${UV_LINK_MODE:-copy}
uv venv "$ENV_DIR" --python "$PYTHON_VERSION"

echo "Installing reproducible PyTorch (CUDA 12.1) and cuDNN..."
uv pip install --python "$ENV_DIR/bin/python" \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.5.1+cu121"

uv pip install --python "$ENV_DIR/bin/python" \
  "nvidia-cudnn-cu12==9.1.0.70"

echo "Installing reproducible JAX (0.4.29) and matching CUDA plugins..."
uv pip install --python "$ENV_DIR/bin/python" \
  "jax==0.4.29" \
  "jaxlib==0.4.29+cuda12.cudnn91" \
  "jax-cuda12-pjrt==0.4.29" \
  "jax-cuda12-plugin==0.4.29" \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  --no-deps

uv pip install --python "$ENV_DIR/bin/python" -e "$ROOT"

if [ "${POOLBASED_WITH_AL4PDE:-1}" = "1" ]; then
  EXTERNAL_ROOT=${POOLBASED_EXTERNAL_ROOT:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/external}
  mkdir -p "$EXTERNAL_ROOT"
  if [ ! -d "$EXTERNAL_ROOT/al4pde/.git" ]; then
    git clone https://github.com/dmusekamp/al4pde.git "$EXTERNAL_ROOT/al4pde"
  fi
  if [ ! -d "$EXTERNAL_ROOT/pdearena/.git" ]; then
    git clone https://github.com/pdearena/pdearena.git "$EXTERNAL_ROOT/pdearena"
  fi
  if [ ! -d "$EXTERNAL_ROOT/jax-cfd/.git" ]; then
    git clone https://github.com/google/jax-cfd.git "$EXTERNAL_ROOT/jax-cfd"
  fi
  git -C "$EXTERNAL_ROOT/al4pde" checkout c183431e43c122717beaf23cbd7f77aebcaaf5c2
  git -C "$EXTERNAL_ROOT/pdearena" checkout 22360a766387c3995220b4a1265a936ab9a81b88
  git -C "$EXTERNAL_ROOT/jax-cfd" checkout d215f13282bd63045fb3455f8fac061653428040
  "$ENV_DIR/bin/python" -m ensurepip --upgrade
  "$ENV_DIR/bin/python" -m pip install tensordict omegaconf tree-math dm-haiku einops gin-config
  "$ENV_DIR/bin/python" -m pip install --no-deps -e "$EXTERNAL_ROOT/jax-cfd" -e "$EXTERNAL_ROOT/pdearena" -e "$EXTERNAL_ROOT/al4pde"
fi

echo "$ENV_DIR"

