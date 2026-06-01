#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NOTEBOOK=${1:-notebooks/loss_generator_round0_lab.ipynb}
NOTEBOOK=$(readlink -f "$NOTEBOOK")

ENV_DIR=${POOLBASED_ENV:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv}
WALLTIME=${BIGFOOT_WALLTIME:-08:00:00}
PROJECT=${BIGFOOT_PROJECT:-pr-melissa}
GPU_MODEL=${BIGFOOT_GPU_MODEL:-any}
CUDA_VERSION=${BIGFOOT_CUDA_VERSION:-12.6}
RUN_DIR=${BIGFOOT_RUN_DIR:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/notebook_sessions}
PORT=${BIGFOOT_NOTEBOOK_PORT:-8899}
SESSION=${BIGFOOT_TMUX_SESSION:-loss_generator_lab}
JOB_NAME=${BIGFOOT_JOB_NAME:-notebook-loss-lab}

mkdir -p "$RUN_DIR"
RUN_DIR=$(readlink -f "$RUN_DIR")
TOKEN=$("$ENV_DIR/bin/python" - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)

WRAPPER="$RUN_DIR/oar-notebook-$(date +%Y%m%d-%H%M%S).sh"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
set +u
source /applis/environments/cuda_env.sh "$CUDA_VERSION"
set -u
source "$ENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export XLA_PYTHON_CLIENT_PREALLOCATE="\${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TORCH_CUDNN_V8_API_ENABLED=0
export CUBLAS_WORKSPACE_CONFIG="\${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export LD_LIBRARY_PATH="$ENV_DIR/lib/python3.11/site-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}"

HOST=\$(hostname -f 2>/dev/null || hostname)
INFO="$RUN_DIR/notebook-\${OAR_JOB_ID:-manual}.info"
LOG="$RUN_DIR/notebook-\${OAR_JOB_ID:-manual}.jupyter.log"
TOKEN_FILE="$RUN_DIR/notebook-\${OAR_JOB_ID:-manual}.token"
echo "$TOKEN" > "\$TOKEN_FILE"

cat > "\$INFO" <<INFO
job_id=\${OAR_JOB_ID:-manual}
node=\$HOST
port=$PORT
tmux_session=$SESSION
notebook=$NOTEBOOK
token_file=\$TOKEN_FILE
url=http://127.0.0.1:$PORT/lab/tree/${NOTEBOOK#$ROOT/}?token=$TOKEN
tunnel=ssh -N -L $PORT:\$HOST:$PORT bigfoot
INFO

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd '$ROOT' && jupyter lab '$NOTEBOOK' --no-browser --ip=0.0.0.0 --port=$PORT --ServerApp.token='$TOKEN' --ServerApp.password='' 2>&1 | tee '\$LOG'"
echo "JupyterLab started in tmux session '$SESSION' on \$HOST:$PORT"
cat "\$INFO"

while tmux has-session -t "$SESSION" 2>/dev/null; do
  sleep 60
done
EOF
chmod +x "$WRAPPER"

OAR_ARGS=(
  -l "/nodes=1/gpu=1,walltime=${WALLTIME}"
  --project "$PROJECT"
  --name "$JOB_NAME"
  -O "$RUN_DIR/%jobid%.out"
  -E "$RUN_DIR/%jobid%.err"
)
if [ "$GPU_MODEL" = "any" ]; then
  OAR_ARGS+=(-p "gpubrand='nvidia'")
elif [ -n "$GPU_MODEL" ]; then
  OAR_ARGS+=(-p "gpumodel='$GPU_MODEL'")
fi

echo "Submitting JupyterLab notebook job"
echo "logs: $RUN_DIR"
oarsub "${OAR_ARGS[@]}" "$WRAPPER"
