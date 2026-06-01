#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET=${1:?Usage: submit_sweep_loss_generator.sh <dataset.npz> <output_dir> [sweep args...]}
OUTPUT_DIR=${2:?Usage: submit_sweep_loss_generator.sh <dataset.npz> <output_dir> [sweep args...]}
shift 2
SWEEP_ARGS=("$@")
SWEEP_ARGS_QUOTED=""
for _arg in "${SWEEP_ARGS[@]+"${SWEEP_ARGS[@]}"}"; do
  SWEEP_ARGS_QUOTED="$SWEEP_ARGS_QUOTED $(printf '%q' "$_arg")"
done

ENV_DIR=${POOLBASED_ENV:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv}
WALLTIME=${BIGFOOT_WALLTIME:-04:00:00}
PROJECT=${BIGFOOT_PROJECT:-pr-melissa}
GPU_MODEL=${BIGFOOT_GPU_MODEL:-any}
CUDA_VERSION=${BIGFOOT_CUDA_VERSION:-12.6}
JOB_NAME=${BIGFOOT_JOB_NAME:-loss-gen-sweep}

DATASET=$(readlink -f "$DATASET")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(readlink -f "$OUTPUT_DIR")

WRAPPER="$OUTPUT_DIR/oar-sweep-$(date +%Y%m%d-%H%M%S).sh"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
set +u
source /applis/environments/cuda_env.sh "$CUDA_VERSION"
set -u
source "$ENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="\${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TORCH_CUDNN_V8_API_ENABLED=0
export CUBLAS_WORKSPACE_CONFIG="\${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export LD_LIBRARY_PATH="$ENV_DIR/lib/python3.11/site-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}"
python -u scripts/sweep_loss_generator.py --dataset "$DATASET" --output-dir "$OUTPUT_DIR"$SWEEP_ARGS_QUOTED
EOF
chmod +x "$WRAPPER"

OAR_ARGS=(
  -l "/nodes=1/gpu=1,walltime=${WALLTIME}"
  --project "$PROJECT"
  --name "$JOB_NAME"
  -O "$OUTPUT_DIR/%jobid%.out"
  -E "$OUTPUT_DIR/%jobid%.err"
)
if [ "$GPU_MODEL" = "any" ]; then
  OAR_ARGS+=(-p "gpubrand='nvidia'")
elif [ -n "$GPU_MODEL" ]; then
  OAR_ARGS+=(-p "gpumodel='$GPU_MODEL'")
fi

oarsub "${OAR_ARGS[@]}" "$WRAPPER"
