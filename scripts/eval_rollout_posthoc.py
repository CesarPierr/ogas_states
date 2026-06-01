#!/usr/bin/env python
"""Post-hoc ROLLOUT evaluation: stable-horizon K_tau (docs/theory.md §9, the *consequence*
of the tube-coverage precondition H1 confirmed in §9.6).

For each run: rebuild the surrogate from config.resolved.json, load the final checkpoint, and
roll it out autoregressively on the uniform validation set. Report per-step NRMSE and the
stable horizon K_tau = first step where NRMSE crosses tau (longer = more stable rollout).

Usage:
    python scripts/eval_rollout_posthoc.py --runs DIR... --uniform <uniform.npz> \
        [--steps 100] [--tau 0.1 0.3 0.5 1.0] [--baseline uniform_baseline] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from poolbased_surrogate.eval import ValidationData, rollout_error_series
from poolbased_surrogate.models.surrogate import build_surrogate


def build_from_cfg(cfg: dict, device):
    s, pde = cfg["surrogate"], cfg["pde"]
    m = build_surrogate(resolution=pde["resolution"], hidden=s["hidden"], depth=s["depth"],
                        ensemble_size=s["ensemble_size"], param_dim=len(pde["param_ranges"]),
                        model_name=s["model"], difference_weight=s.get("difference_weight", 0.3)).to(device)
    m.eval()
    return m


def load_validation(npz_path: Path) -> ValidationData:
    with np.load(npz_path) as d:
        return ValidationData(states0=d["states0"], params=d["params"], trajectories=d["trajectories"])


def k_tau(series: np.ndarray, steps: np.ndarray, tau: float) -> float:
    """First step index where the per-step error first exceeds tau (else max horizon)."""
    over = np.where(series > tau)[0]
    return float(steps[over[0]]) if len(over) else float(steps[-1])


def _ms(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--uniform", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--tau", nargs="+", type=float, default=[0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--n_traj", type=int, default=512)
    ap.add_argument("--baseline", default="uniform_baseline")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    val = load_validation(Path(args.uniform))
    steps = min(args.steps, val.trajectories.shape[1] - 1)

    rx = re.compile(r"(.+)_seed(\d+)$")
    agg = defaultdict(list)
    for run in args.runs:
        run = Path(run)
        m = rx.match(run.name)
        cfgp, ckptp = run / "config.resolved.json", run / "checkpoint_latest.pt"
        if not (m and cfgp.exists() and ckptp.exists()):
            print(f"  skip {run.name}", flush=True)
            continue
        variant = m.group(1)
        model = build_from_cfg(json.loads(cfgp.read_text()), device)
        try:
            ck = torch.load(ckptp, map_location=device, weights_only=False, mmap=True)
        except TypeError:
            ck = torch.load(ckptp, map_location=device, weights_only=False)
        model.load_state_dict(ck["surrogate"])
        ser = rollout_error_series(model, val, steps=steps, device=device, n_trajectories=args.n_traj)
        rec = {"nrmse@10": float(ser["nrmse_mean"][min(9, steps - 1)]),
               "nrmse@50": float(ser["nrmse_mean"][min(49, steps - 1)]),
               "nrmse@final": float(ser["nrmse_mean"][-1])}
        for t in args.tau:
            rec[f"Ktau_mean_{t}"] = k_tau(ser["nrmse_mean"], ser["step"], t)
            rec[f"Ktau_p95_{t}"] = k_tau(ser["nrmse_p95"], ser["step"], t)
        agg[variant].append(rec)
        print(f"  {variant} seed{m.group(2)}: nrmse@final={rec['nrmse@final']:.3f}", flush=True)

    variants = sorted(agg.keys())
    fields = (["nrmse@10", "nrmse@50", "nrmse@final"]
              + [f"Ktau_mean_{t}" for t in args.tau] + [f"Ktau_p95_{t}" for t in args.tau])
    print("\n=== ROLLOUT (mean over seeds). nrmse: LOWER=better; Ktau: HIGHER=better (longer stable horizon) ===")
    hdr = f"{'variant':<20}" + "".join(f"{f.replace('Ktau_',''):>14}" for f in fields)
    print(hdr); print("-" * len(hdr))
    base = {}
    for v in variants:
        row = f"{v:<20}"
        for f in fields:
            mn, _ = _ms([r[f] for r in agg[v]])
            if v == args.baseline:
                base[f] = mn
            row += f"{mn:>14.3f}"
        print(row)
    if args.baseline in agg:
        print(f"\n  vs '{args.baseline}': nrmse %change (neg=better) | Ktau ratio (>1=better)")
        print(hdr); print("-" * len(hdr))
        for v in variants:
            if v == args.baseline:
                continue
            row = f"{v:<20}"
            for f in fields:
                mn, _ = _ms([r[f] for r in agg[v]])
                bm = base.get(f)
                if f.startswith("nrmse"):
                    val_ = (mn - bm) / bm * 100 if bm else float("nan")
                    row += f"{val_:>+13.1f}%"
                else:
                    row += f"{(mn / bm if bm else float('nan')):>13.2f}x"
            print(row)


if __name__ == "__main__":
    main()
