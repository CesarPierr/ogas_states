#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?Usage: submit_bigfoot.sh <config.yaml>}
CONFIG=$(readlink -f "$CONFIG")
shift
RUN_ARGS=("$@")
RUN_ARGS_QUOTED=$(printf " %q" "${RUN_ARGS[@]}")

ENV_DIR=${POOLBASED_ENV:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv}
WALLTIME=${BIGFOOT_WALLTIME:-08:00:00}
GPUS=${BIGFOOT_NUM_GPUS:-1}
GPU_MODEL=${BIGFOOT_GPU_MODEL:-A100}
PROJECT=${BIGFOOT_PROJECT:-pr-melissa}
JOB_NAME=${BIGFOOT_JOB_NAME:-poolbased-bigfoot}
RUN_DIR=${BIGFOOT_RUN_DIR:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_oar}
OAR_TYPE=${BIGFOOT_OAR_TYPE:-}
CUDA_VERSION=${BIGFOOT_CUDA_VERSION:-12.6}
WANDB_MODE_VALUE=${WANDB_MODE:-online}
PRECREATE_VALIDATION=${BIGFOOT_PRECREATE_VALIDATION:-1}
mkdir -p "$RUN_DIR"

if [ "$PRECREATE_VALIDATION" = "1" ]; then
  "$ENV_DIR/bin/python" -m poolbased_surrogate.run "$CONFIG" "${RUN_ARGS[@]}" --create-validation-only
fi

RESOURCE="/nodes=1/gpu=${GPUS},walltime=${WALLTIME}"
if [ "$OAR_TYPE" = "devel" ]; then
  RESOURCE="/nodes=1/gpu=${GPUS}/migdevice=1,walltime=${WALLTIME}"
fi

WRAPPER="$RUN_DIR/oar-$(basename "${CONFIG%.*}")-$(date +%Y%m%d-%H%M%S).sh"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
set +u
source /applis/environments/cuda_env.sh "$CUDA_VERSION"
set -u
source "$ENV_DIR/bin/activate"
export WANDB_MODE="$WANDB_MODE_VALUE"
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-4}"
export XLA_PYTHON_CLIENT_PREALLOCATE="\${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TORCH_CUDNN_V8_API_ENABLED=0
python -m poolbased_surrogate.run "$CONFIG" --resume$RUN_ARGS_QUOTED
EOF
chmod +x "$WRAPPER"

OAR_ARGS=(
  -l "$RESOURCE"
  -p "gpumodel='${GPU_MODEL}'"
  --project "$PROJECT"
  --name "$JOB_NAME"
  -O "$RUN_DIR/%jobid%.out"
  -E "$RUN_DIR/%jobid%.err"
)
if [ -n "$OAR_TYPE" ]; then
  OAR_ARGS=(-t "$OAR_TYPE" "${OAR_ARGS[@]}")
fi

oarsub "${OAR_ARGS[@]}" "$WRAPPER"
