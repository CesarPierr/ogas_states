#!/usr/bin/env bash
#SBATCH --job-name=eval_master_checkpoints
#SBATCH --account=EUHPC_D36_033
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=results/eval_master_checkpoints_%j.out
#SBATCH --error=results/eval_master_checkpoints_%j.err

set -euo pipefail

cd /leonardo/home/userexternal/pcesar00/ogas_states
source .venv/bin/activate

mkdir -p results/json

echo "=== STARTING MASTER POST-HOC EVALUATION ON GPU ==="
echo "Node: $(hostname)"
echo "CUDA Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

python -u scripts/eval_all_checkpoints_master.py

echo "=== MASTER POST-HOC EVALUATION COMPLETE ==="
