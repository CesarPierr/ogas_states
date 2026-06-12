#!/usr/bin/env bash
# SLURM submitter for ad-hoc loss-generator-lab jobs on Leonardo (CINECA) — mirrors submit_leonardo.sh.
#
# Usage:  ./scripts/submit_lab_leonardo.sh <walltime> -- <python args...>
#   e.g.  ./scripts/submit_lab_leonardo.sh 04:00:00 -- scripts/train_loss_generator.py \
#             --dataset "$FAST/ogas_states_runs/lab/ds.npz" --output-dir "$FAST/ogas_states_runs/lab/fm_s64"
#   e.g.  ./scripts/submit_lab_leonardo.sh 12:00:00 -- scripts/sweep_loss_generator.py \
#             --dataset "$FAST/ogas_states_runs/lab/ds.npz" --output-dir "$FAST/ogas_states_runs/lab/sweep1"
# Env:
#   LEONARDO_ACCOUNT   Slurm account            (default euhpc_d36_033 — CHANGE if needed)
#   LEONARDO_QOS       qos                      (default normal)
#   LEONARDO_PARTITION partition                (default boost_usr_prod)
#   LAB_JOB_NAME       job name                 (default lg-lab)
#   LAB_RUN_DIR        output dir for logs      (default $FAST/ogas_states_runs/lab_adhoc)
#   POOLBASED_ENV      venv dir                 (default <repo>/.venv, wherever the repo lives)
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WALLTIME=${1:?Usage: submit_lab_leonardo.sh <walltime> -- <python args...>}
shift
if [ "${1:-}" != "--" ]; then
  echo "Usage: submit_lab_leonardo.sh <walltime> -- <python args...>" >&2
  exit 1
fi
shift
if [ "$#" -eq 0 ]; then
  echo "Usage: submit_lab_leonardo.sh <walltime> -- <python args...>" >&2
  exit 1
fi
PY_ARGS_QUOTED=""
for _arg in "$@"; do
  PY_ARGS_QUOTED="$PY_ARGS_QUOTED $(printf '%q' "$_arg")"
done

ENV_DIR=${POOLBASED_ENV:-$ROOT/.venv}
ACCOUNT=${LEONARDO_ACCOUNT:-euhpc_d36_033}
QOS=${LEONARDO_QOS:-normal}
PARTITION=${LEONARDO_PARTITION:-boost_usr_prod}
JOB_NAME=${LAB_JOB_NAME:-lg-lab}
RUN_DIR=${LAB_RUN_DIR:-$FAST/ogas_states_runs/lab_adhoc}
mkdir -p "$RUN_DIR"

WRAPPER="$RUN_DIR/slurm-lab-$(date +%Y%m%d-%H%M%S).sbatch"
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
python -u$PY_ARGS_QUOTED
EOF

sbatch "$WRAPPER"
