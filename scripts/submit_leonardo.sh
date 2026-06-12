#!/usr/bin/env bash
# SLURM submitter for Leonardo (CINECA) booster partition — mirrors submit_bigfoot.sh.
#
# Usage:  ./scripts/submit_leonardo.sh <config.yaml> [run args...]
# Env:
#   LEONARDO_ACCOUNT   Slurm account            (default euhpc_d36_033 — CHANGE if needed)
#   LEONARDO_QOS       qos                      (default normal)
#   LEONARDO_PARTITION partition                (default boost_usr_prod)
#   LEONARDO_WALLTIME  walltime HH:MM:SS        (default 08:00:00; boost normal cap is 24h)
#   LEONARDO_JOB_NAME  job name                 (default poolbased-leo)
#   LEONARDO_RUN_DIR   output dir for logs      (default $FAST/ogas_states_runs/adhoc)
#   POOLBASED_ENV      venv dir                 (default <repo>/.venv, wherever the repo lives)
#   LEONARDO_PRECREATE_VALIDATION  1|0          (default 1; runs --create-validation-only on login node first)
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?Usage: submit_leonardo.sh <config.yaml>}
CONFIG=$(readlink -f "$CONFIG")
shift
RUN_ARGS=("$@")
RUN_ARGS_QUOTED=""
for _arg in "${RUN_ARGS[@]+"${RUN_ARGS[@]}"}"; do
  RUN_ARGS_QUOTED="$RUN_ARGS_QUOTED $(printf '%q' "$_arg")"
done

ENV_DIR=${POOLBASED_ENV:-$ROOT/.venv}
ACCOUNT=${LEONARDO_ACCOUNT:-euhpc_d36_033}
QOS=${LEONARDO_QOS:-normal}
PARTITION=${LEONARDO_PARTITION:-boost_usr_prod}
WALLTIME=${LEONARDO_WALLTIME:-08:00:00}
JOB_NAME=${LEONARDO_JOB_NAME:-poolbased-leo}
RUN_DIR=${LEONARDO_RUN_DIR:-$FAST/ogas_states_runs/adhoc}
PRECREATE_VALIDATION=${LEONARDO_PRECREATE_VALIDATION:-1}
mkdir -p "$RUN_DIR"

if [ "$PRECREATE_VALIDATION" = "1" ]; then
  JAX_PLATFORMS=cpu "$ENV_DIR/bin/python" -m poolbased_surrogate.run "$CONFIG" "${RUN_ARGS[@]}" --create-validation-only
fi

WRAPPER="$RUN_DIR/slurm-$(basename "${CONFIG%.*}")-$(date +%Y%m%d-%H%M%S).sbatch"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
#SBATCH --account=$ACCOUNT
#SBATCH --qos=$QOS
#SBATCH --partition=$PARTITION
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=$WALLTIME
#SBATCH --output=$RUN_DIR/%j.out
#SBATCH --error=$RUN_DIR/%j.err
set -euo pipefail
cd "$ROOT"
module purge
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
source "$ENV_DIR/bin/activate"
# Leonardo compute nodes have no outbound internet: online wandb.init() blocks
# forever. Default to offline (stdout JSON + history.json always written);
# sync later from the login node with: wandb sync wandb/offline-run-*
export WANDB_MODE=\${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-8}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCH_CUDNN_V8_API_ENABLED=0
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$ENV_DIR/lib/python3.11/site-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}"
python -u -m poolbased_surrogate.run "$CONFIG" --resume$RUN_ARGS_QUOTED
EOF

sbatch "$WRAPPER"
