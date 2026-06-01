#!/usr/bin/env python
"""Generate a validation dataset for the sub5 KS scenario.

Uses n_substeps=5 (dt_inner=0.01) and n_warmup=20 so that the recorded
trajectories start from a point already on the attractor and each outer step
is numerically stable despite the larger apparent dt=0.05.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from poolbased_surrogate.config import PDEConfig
from poolbased_surrogate.eval import ValidationData, create_validation_data, save_validation_data
from poolbased_surrogate.pde import PDE1D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-trajectories", type=int, default=1500)
    parser.add_argument("--trajectory-steps", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--domain-length", type=float, default=64.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-substeps", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--param1-min", type=float, default=10.0)
    parser.add_argument("--param1-max", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    cfg = PDEConfig(
        name="ks",
        backend="al4pde",
        resolution=args.resolution,
        domain_length=args.domain_length,
        dt=args.dt,
        param_ranges=[[0.5, 4.0], [args.param1_min, args.param1_max]],
        param_log_scale=[False, False],
        ic={"N": 10, "lmin": 1, "lmax": 10, "amplitude": 1.0},
        n_substeps=args.n_substeps,
        n_warmup=args.n_warmup,
    )
    pde = PDE1D(cfg)

    print(f"Generating {args.n_trajectories} trajectories × {args.trajectory_steps} steps")
    print(f"  solver dt={args.dt}  n_substeps={args.n_substeps}  transition_dt={args.dt*args.n_substeps:.3f}")
    print(f"  n_warmup={args.n_warmup}  seed={args.seed}")

    params = pde.sample_params_halton(args.n_trajectories, seed=args.seed)
    states0 = pde.sample_ic_halton(args.n_trajectories, seed=args.seed + 1000)

    # Simulate in batches so we can track progress and handle failures gracefully
    all_trajs = []
    valid_mask = np.ones(args.n_trajectories, dtype=bool)
    for start in range(0, args.n_trajectories, args.batch_size):
        end = min(start + args.batch_size, args.n_trajectories)
        print(f"  batch {start}:{end} ...", flush=True)
        try:
            traj = pde.simulate(states0[start:end], params[start:end], args.trajectory_steps, apply_warmup=True)
        except Exception as exc:
            print(f"  WARNING batch {start}:{end} failed: {exc}")
            traj = np.full((end - start, args.trajectory_steps + 1, 1, args.resolution), np.nan, dtype=np.float32)
            valid_mask[start:end] = False
        finite = np.isfinite(traj).all(axis=(1, 2, 3))
        if not finite.all():
            print(f"  WARNING: {(~finite).sum()} trajectories have NaN in batch {start}:{end}")
            valid_mask[start:end] &= finite
        all_trajs.append(traj.astype(np.float32))

    trajectories = np.concatenate(all_trajs, axis=0)
    n_valid = int(valid_mask.sum())
    print(f"\nValid trajectories: {n_valid}/{args.n_trajectories}")
    print(f"State stats: mean={trajectories[valid_mask].mean():.4f}  std={trajectories[valid_mask].std():.4f}")
    print(f"  abs p95={np.quantile(np.abs(trajectories[valid_mask]), 0.95):.3f}")
    print(f"  total_variation mean={np.mean(np.sum(np.abs(np.diff(trajectories[valid_mask, 0], axis=-1)), axis=-1)):.1f}")

    data = ValidationData(
        states0=states0,
        params=params,
        trajectories=trajectories,
    )
    out = Path(args.output)
    save_validation_data(out, data)
    # Also save metadata
    import json
    (out.with_suffix(".metadata.json")).write_text(json.dumps({
        "seed": args.seed,
        "n_trajectories": args.n_trajectories,
        "trajectory_steps": args.trajectory_steps,
        "resolution": args.resolution,
        "domain_length": args.domain_length,
        "dt": args.dt,
        "n_substeps": args.n_substeps,
        "n_warmup": args.n_warmup,
        "n_valid": n_valid,
        "pde": "KS (KSVarLVIC)",
        "param_ranges": [[0.5, 4.0], [0.1, 100.0]],
    }, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
