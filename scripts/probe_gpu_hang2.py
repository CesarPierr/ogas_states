"""Full-scale reproduction of the pilot first solver call via the bridge."""
import time

import numpy as np

def stage(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

stage("0: imports")
import torch
from poolbased_surrogate.config import load_config
from poolbased_surrogate.pde import PDE1D

cfg = load_config("configs/leonardo_ks_v3_base.yaml")
pde = PDE1D(cfg.pde)
rng = np.random.default_rng(101)
stage("1: bridge ready (KS res800 sub5 warmup20)")

# mimic run.py: torch model on GPU first
dummy = torch.zeros(4096, 4096, device="cuda")
torch.cuda.synchronize()
stage("2: torch CUDA context up")

params = pde.sample_params_uniform(250, rng)
states0 = pde.sample_ic_uniform(250, rng)
stage(f"3: ICs sampled {states0.shape}; calling simulate(steps=70) [warmup 20 + main 70]")
traj = pde.simulate(states0, params, 70)
stage(f"4: simulate done, traj {traj.shape}, finite={np.isfinite(traj).all()}")
