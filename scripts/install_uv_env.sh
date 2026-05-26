#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${POOLBASED_ENV:-/hoyt/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv}
PYTHON_VERSION=${POOLBASED_PYTHON:-3.11}

uv venv "$ENV_DIR" --python "$PYTHON_VERSION"
uv pip install --python "$ENV_DIR/bin/python" -e "$ROOT"

echo "$ENV_DIR"

