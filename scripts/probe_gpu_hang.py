"""Staged GPU probe to localize the pilot-job hang. Run under srun --overlap."""
import os, sys, time

def stage(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

stage("0: start")
import torch
stage(f"1: torch {torch.__version__} cuda={torch.cuda.is_available()}")
t = torch.zeros(1024, device="cuda")
torch.cuda.synchronize()
stage("2: torch cuda tensor OK")

import jax
stage(f"3: jax {jax.__version__} imported")
stage(f"4: jax devices: {jax.devices()}")

import jax.numpy as jnp
x = jnp.ones((4, 800))
y = jnp.fft.rfft(x).block_until_ready()
stage("5: jax rfft on GPU OK")

import numpy as np
sys.path.insert(0, os.environ["AL4PDE_ROOT"])
sys.path.insert(0, os.environ["JAX_CFD_ROOT"])
from al4pde.tasks.sim.ks_jax import build_jax_sim
stage("6: build_jax_sim imported")
sim = build_jax_sim(800, 0.05, 2)
u0 = np.random.default_rng(0).standard_normal((1, 2, 800)).astype(np.float32)
par = np.array([[[1.0, 64.0]] * 2], dtype=np.float32)
stage("7: calling sim (compile + run, nt=2, batch=2)")
out = sim((u0, par))
out = np.asarray(out)
stage(f"8: sim done, out shape {out.shape}, finite={np.isfinite(out).all()}")
