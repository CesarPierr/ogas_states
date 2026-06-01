#!/usr/bin/env python
"""Build the TUBE validation set (docs/theory.md §8.4 / §9.5): attractor states perturbed
*off* the trajectory manifold, then stepped through the real solver.

These states are genuinely hard in a way the TV / low-amplitude tails are not:
 - they are OFF-attractor (the surrogate, trained on uniform trajectories, never saw them),
 - they have meaningful amplitude (no NRMSE denominator artifact like the low-amp tail),
 - they are exactly the kind of perturbed states a rollout drifts into (exposure bias, §9).

Perturbation: structured low-wavenumber field of RMS = rho * RMS(u), added to attractor
states; rho sweeps the tube radius. Output npz schema matches the uniform validation set
(states0, params, trajectories=[perturbed_state, true_next]).

Usage:
    python scripts/build_tube_validation.py <config.(yaml|resolved.json)> <uniform.npz> <out_dir> \
        [--rho 0.1 0.25 0.5] [--n 4000] [--kmax 12] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poolbased_surrogate.config import PDEConfig, load_config
from poolbased_surrogate.pde import PDE1D


def load_pde_cfg(path: str) -> PDEConfig:
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())["pde"]
        return PDEConfig(**d)
    return load_config(path).pde


def smooth_perturbation(n: int, L: int, kmax: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS real fields with energy in wavenumbers 1..kmax (periodic, KS-like)."""
    x = np.linspace(0, 2 * np.pi, L, endpoint=False)
    field = np.zeros((n, L), dtype=np.float64)
    for k in range(1, kmax + 1):
        a = rng.normal(size=(n, 1)) / k
        b = rng.normal(size=(n, 1)) / k
        field += a * np.cos(k * x)[None, :] + b * np.sin(k * x)[None, :]
    field -= field.mean(axis=1, keepdims=True)
    field /= (np.sqrt(np.mean(field ** 2, axis=1, keepdims=True)) + 1e-12)
    return field.astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("uniform_npz")
    ap.add_argument("out_dir")
    ap.add_argument("--rho", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver_batch", type=int, default=512)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pde = PDE1D(load_pde_cfg(args.config))

    with np.load(args.uniform_npz) as d:
        traj, params = d["trajectories"], d["params"]
    n_traj, tp1, _, L = traj.shape
    # attractor states: interior/late frames (well past warmup), one per sampled (traj, frame)
    flat = traj[:, tp1 // 2:].reshape(-1, L)            # late-time states, on attractor
    flat_par = np.repeat(params, tp1 - tp1 // 2, axis=0)
    pick = rng.choice(len(flat), size=min(args.n, len(flat)), replace=False)
    base = flat[pick].astype(np.float32)
    par = flat_par[pick].astype(np.float32)
    base_rms = np.sqrt(np.mean(base ** 2, axis=1, keepdims=True))
    print(f"bank attractor states: {len(flat)}; sampled {len(base)} (L={L}); base RMS med={np.median(base_rms):.3f}")

    for rho in args.rho:
        delta = smooth_perturbation(len(base), L, args.kmax, rng) * (rho * base_rms)
        pert = (base + delta).astype(np.float32)
        # true next state of the PERTURBED state via the real solver (no warmup)
        nxt = np.empty_like(pert)
        for i in range(0, len(pert), args.solver_batch):
            j = min(i + args.solver_batch, len(pert))
            nxt[i:j] = pde.step(pert[i:j], par[i:j])
        finite = np.isfinite(nxt).all(axis=1) & np.isfinite(pert).all(axis=1)
        pert, nxt, parr = pert[finite], nxt[finite], par[finite]
        s0 = pert[:, None, :]
        traj_out = np.stack([pert, nxt], axis=1)[:, :, None, :].astype(np.float32)  # (N,2,1,L)
        tag = f"{rho}".replace(".", "p")
        path = out / f"val_tube_rho{tag}.npz"
        np.savez_compressed(path, states0=s0.astype(np.float32), params=parr.astype(np.float32),
                            trajectories=traj_out)
        print(f"  wrote {path}  ({len(pert)} states, finite={finite.mean():.3f}, "
              f"off-manifold RMS shift = {rho:.2f}x)")


if __name__ == "__main__":
    main()
