#!/bin/bash
#OAR -l /nodes=1/gpu=1/migdevice=1,walltime=00:05:00
#OAR -O test_jax_torch.out
#OAR -E test_jax_torch.err
#OAR -t devel
set -euo pipefail

set +u
source /applis/environments/cuda_env.sh 12.6
set -u

source /bettik/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv/bin/activate

export LD_LIBRARY_PATH="/bettik/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "=== PyTorch Test ==="
python -c "import torch; import torch.nn as nn; conv = nn.Conv1d(1, 1, 3).cuda(); x = torch.randn(1, 1, 10).cuda(); print('Torch Conv Output Shape:', conv(x).shape); print('Torch GPU status: SUCCESS')"

echo "=== JAX Test ==="
python -c "import jax; import jax.numpy as jnp; print('JAX devices:', jax.devices()); x = jnp.ones((3, 3)); print('JAX calculation:', (x + x).sum()); print('JAX status: SUCCESS')"
