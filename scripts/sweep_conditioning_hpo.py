#!/usr/bin/env python
"""Conditioning-HPO sweep: can CFG / FiLM / embedding size make difficulty
conditioning actually steer realized difficulty? (pilot verdict: input-concat
conditioning is dead — unselected spearman -0.43 at round 9, steering is 100%
oversample-reject selection.)

Backbone fixed to the production generator (hidden=64, steps=64, rb=2, k=7,
n_quantiles=20); arms vary only the conditioning mechanism. One arm per SLURM
job via --arm; score_metrics.json per arm (skip_existing allows relaunch).

  python scripts/sweep_conditioning_hpo.py --dataset DS.npz --output-dir OUT [--arm NAME]
"""
from __future__ import annotations

import argparse

from poolbased_surrogate.lab import run_sweep

BB = {"generator": "flow_matching", "hidden": 64, "steps": 64, "residual_blocks": 2,
      "kernel_size": 7, "n_quantiles": 20}

ARMS = [
    # control = exactly the production conditioning path
    {**BB, "name": "ctrl_concat"},
    # classifier-free guidance on the difficulty bin (training dropout x sampling scale)
    {**BB, "name": "cfg_d10_s15", "cfg_dropout": 0.1, "cfg_scale": 1.5},
    {**BB, "name": "cfg_d10_s3",  "cfg_dropout": 0.1, "cfg_scale": 3.0},
    {**BB, "name": "cfg_d10_s5",  "cfg_dropout": 0.1, "cfg_scale": 5.0},
    {**BB, "name": "cfg_d20_s3",  "cfg_dropout": 0.2, "cfg_scale": 3.0},
    # FiLM per-block injection (vs input concat)
    {**BB, "name": "film",            "cond_mode": "film"},
    {**BB, "name": "film_cfg_d10_s3", "cond_mode": "film", "cfg_dropout": 0.1, "cfg_scale": 3.0},
    {**BB, "name": "film_cfg_d10_s5", "cond_mode": "film", "cfg_dropout": 0.1, "cfg_scale": 5.0},
    # bigger label embedding (default is max(8, hidden//8) = 8 at hidden=64)
    {**BB, "name": "qed32",            "quant_embed_dim": 32},
    {**BB, "name": "qed32_cfg_d10_s3", "quant_embed_dim": 32, "cfg_dropout": 0.1, "cfg_scale": 3.0},
    {**BB, "name": "film_qed32_cfg_d10_s3", "cond_mode": "film", "quant_embed_dim": 32,
     "cfg_dropout": 0.1, "cfg_scale": 3.0},
    # coarser bins: 10 instead of 20 (more samples per bin -> stronger signal?)
    {**BB, "name": "nq10_cfg_d10_s3", "n_quantiles": 10, "cfg_dropout": 0.1, "cfg_scale": 3.0},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--arm", default=None, help="run a single arm by name (default: all)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    configs = ARMS if args.arm is None else [a for a in ARMS if a["name"] == args.arm]
    if not configs:
        raise SystemExit(f"unknown arm {args.arm!r}; choices: {[a['name'] for a in ARMS]}")
    run_sweep(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        configs=configs,
        n_quantiles=20,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
