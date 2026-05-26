#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?Usage: submit_kraken_devel.sh <config.yaml>}
CONFIG=$(readlink -f "$CONFIG")
ENV_DIR=${POOLBASED_ENV:-/hoyt/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv}
WALLTIME=${KRAKEN_WALLTIME:-00:30:00}
GPUS=${KRAKEN_NUM_GPUS:-1}
PROJECT=${KRAKEN_PROJECT:-pr-melissa}
JOB_NAME=${KRAKEN_JOB_NAME:-poolbased-devel}
RUN_DIR=${KRAKEN_RUN_DIR:-/hoyt/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_oar}
OAR_TYPE=${KRAKEN_OAR_TYPE:-devel}
mkdir -p "$RUN_DIR"

WRAPPER="$RUN_DIR/oar-$(basename "${CONFIG%.*}")-$(date +%Y%m%d-%H%M%S).sh"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
source "$ENV_DIR/bin/activate"
export WANDB_MODE="\${WANDB_MODE:-online}"
python -m poolbased_surrogate.run "$CONFIG" --resume
EOF
chmod +x "$WRAPPER"

OAR_ARGS=(
  -l "/nodes=1/gpu=${GPUS},walltime=${WALLTIME}"
  --project "$PROJECT"
  --name "$JOB_NAME"
  -O "$RUN_DIR/%jobid%.out"
  -E "$RUN_DIR/%jobid%.err"
)
if [ -n "$OAR_TYPE" ]; then
  OAR_ARGS=(-t "$OAR_TYPE" "${OAR_ARGS[@]}")
fi

oarsub \
  "${OAR_ARGS[@]}" \
  "$WRAPPER"
