#!/usr/bin/env python
"""Post-hoc test of docs/theory.md §8.5 on ALREADY-TRAINED runs.

For each run dir: rebuild the surrogate from config.resolved.json, load the final checkpoint
weights, and evaluate one-step NRMSE on (a) the uniform validation set and (b) the HARD /
COVERAGE sets built by build_hard_validation.py. The hypothesis: generator (hard-mining)
variants beat the uniform baseline by a *much larger* margin on val_hard than on val_uniform,
and they *flatten the loss profile* (lower sup/mean and Gini on the coverage set).

Usage:
    python scripts/eval_hard_posthoc.py --runs DIR1 DIR2 ... \
        --uniform <uniform.npz> --hard <hard_dir> [--device cuda]
A run dir is <base>/<variant>_seed<seed>; variants/seeds are parsed from the names.
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

from poolbased_surrogate.models.surrogate import build_surrogate


def load_transitions(npz_path: Path):
    with np.load(npz_path) as d:
        traj, params = d["trajectories"], d["params"]
    n, tp1 = traj.shape[:2]
    steps = tp1 - 1
    state = traj[:, :-1].reshape(n * steps, *traj.shape[2:]).astype(np.float32)
    nxt = traj[:, 1:].reshape(n * steps, *traj.shape[2:]).astype(np.float32)
    par = np.repeat(params, steps, axis=0).astype(np.float32)
    return state, nxt, par


@torch.no_grad()
def errors(model, state, nxt, par, device, bs=1024):
    """Return (rmse_abs, nrmse) per transition."""
    r_out, n_out = [], []
    for i in range(0, len(state), bs):
        s = torch.from_numpy(state[i:i + bs]).to(device)
        p = torch.from_numpy(par[i:i + bs]).to(device)
        t = torch.from_numpy(nxt[i:i + bs]).to(device)
        pred = model(s, p)
        rmse = torch.sqrt(torch.mean((pred - t) ** 2, dim=(1, 2)))
        denom = torch.sqrt(torch.mean(t ** 2, dim=(1, 2))) + 1e-8
        r_out.append(rmse.cpu().numpy())
        n_out.append((rmse / denom).cpu().numpy())
    return np.concatenate(r_out), np.concatenate(n_out)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def build_from_cfg(cfg: dict, device):
    s = cfg["surrogate"]
    pde = cfg["pde"]
    model = build_surrogate(
        resolution=pde["resolution"],
        hidden=s["hidden"],
        depth=s["depth"],
        ensemble_size=s["ensemble_size"],
        param_dim=len(pde["param_ranges"]),
        model_name=s["model"],
        difference_weight=s.get("difference_weight", 0.3),
    ).to(device)
    model.eval()
    return model


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
    ap.add_argument("--hard", required=True, help="dir with val_hard_*.npz and val_cov.npz")
    ap.add_argument("--baseline", default="uniform")
    ap.add_argument("--uniform_sub", type=int, default=20000)
    ap.add_argument("--cap", type=int, default=0, help="cap every set to this many transitions (0=off)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    harddir = Path(args.hard)

    # Validation sets
    sets = {"val_unif": load_transitions(Path(args.uniform))}
    if args.uniform_sub and len(sets["val_unif"][0]) > args.uniform_sub:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(sets["val_unif"][0]), args.uniform_sub, replace=False)
        sets["val_unif"] = tuple(a[idx] for a in sets["val_unif"])
    for f in sorted(harddir.glob("val_hard_*.npz")) + sorted(harddir.glob("val_tube_*.npz")) + [harddir / "val_cov.npz"]:
        if f.exists():
            sets[f.stem] = load_transitions(f)
    if args.cap:
        rng = np.random.default_rng(1)
        for k, v in list(sets.items()):
            if len(v[0]) > args.cap:
                idx = rng.choice(len(v[0]), args.cap, replace=False)
                sets[k] = tuple(a[idx] for a in v)
    print("validation sets:", {k: len(v[0]) for k, v in sets.items()}, flush=True)

    rx = re.compile(r"(.+)_seed(\d+)$")
    # variant -> setname -> list of dicts
    agg: dict = defaultdict(lambda: defaultdict(list))
    for run in args.runs:
        run = Path(run)
        m = rx.match(run.name)
        if not m:
            continue
        variant, seed = m.group(1), int(m.group(2))
        cfgp, ckptp = run / "config.resolved.json", run / "checkpoint_latest.pt"
        if not (cfgp.exists() and ckptp.exists()):
            print(f"  skip {run.name} (missing files)")
            continue
        cfg = json.loads(cfgp.read_text())
        model = build_from_cfg(cfg, device)
        try:
            ck = torch.load(ckptp, map_location=device, weights_only=False, mmap=True)
        except TypeError:
            ck = torch.load(ckptp, map_location=device, weights_only=False)
        model.load_state_dict(ck["surrogate"])
        nrounds = ck.get("next_round")
        for name, (st, nx, pr) in sets.items():
            r, e = errors(model, st, nx, pr, device)
            rec = {"rmse": float(r.mean()), "rmse_p99": float(np.quantile(r, 0.99)),
                   "mean": float(e.mean()), "p95": float(np.quantile(e, 0.95)),
                   "p99": float(np.quantile(e, 0.99)), "sup_mean": float(e.max() / (e.mean() + 1e-12)),
                   "gini": gini(e), "_rounds": nrounds}
            agg[variant][name].append(rec)
        print(f"  {variant} seed{seed}: rounds={nrounds} done", flush=True)

    setnames = list(sets.keys())
    variants = sorted(agg.keys())

    def table(field, title):
        print(f"\n=== {title} (mean over seeds; LOWER=better) ===")
        hdr = f"{'variant':<22}" + "".join(f"{s:>14}" for s in setnames)
        print(hdr); print("-" * len(hdr))
        base = {}
        for v in variants:
            row = f"{v:<22}"
            for s in setnames:
                mn, _ = _ms([r[field] for r in agg[v][s]])
                if v == args.baseline:
                    base[s] = mn
                row += f"{mn:>14.4f}"
            print(row)
        if args.baseline in agg:
            print(f"\n  % change vs '{args.baseline}' (neg=better):")
            print(hdr); print("-" * len(hdr))
            for v in variants:
                if v == args.baseline:
                    continue
                row = f"{v:<22}"
                for s in setnames:
                    mn, _ = _ms([r[field] for r in agg[v][s]])
                    bm = base.get(s)
                    pct = (mn - bm) / bm * 100 if bm and math.isfinite(mn) and bm else float("nan")
                    row += f"{pct:>+13.1f}%"
                print(row)

    table("rmse", "ONE-STEP RMSE mean (ABSOLUTE; bulk-dominated)")
    table("rmse_p99", "ONE-STEP RMSE p99 (ABSOLUTE)")
    table("mean", "ONE-STEP NRMSE mean (RELATIVE)")
    table("p99", "ONE-STEP NRMSE p99 (RELATIVE)")
    table("sup_mean", "LOSS-PROFILE INEQUALITY  sup/mean  (flatter=better coverage)")
    table("gini", "LOSS-PROFILE INEQUALITY  Gini")


if __name__ == "__main__":
    main()
