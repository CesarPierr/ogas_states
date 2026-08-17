#!/usr/bin/env python
"""Detailed post-hoc test on ALREADY-TRAINED runs.

Computes transition-wise errors and aggregates them into statistical metrics
(mean, std, min, max, p10, p25, p50, p75, p90, p95, p99) for both absolute RMSE
and relative NRMSE, then performs Welch's t-test and Mann-Whitney U test vs baselines.

Usage:
    python scripts/eval_campaign_posthoc_detailed.py \
        --base-dir $FAST/ogas_states_runs/ks800_v3_pilot \
        --uniform $FAST/ogas_validation/ks_res800_al4pde_sub5_seed43_n1500_t100.npz \
        --hard $FAST/ogas_validation/ks_diverse_suite
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu, ttest_ind

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
def calculate_errors(model, state, nxt, par, device, bs=1024):
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=None,
                    help="Directory containing run folders")
    ap.add_argument("--runs", nargs="+", default=[],
                    help="Explicit list of run folders (ignores base-dir if set)")
    ap.add_argument("--uniform", required=True, help="Uniform validation npz path")
    ap.add_argument("--hard", required=True, help="Directory containing diverse hard validation sets")
    ap.add_argument("--baselines", default="uniform_baseline,random_tube",
                    help="Comma-separated baselines to compare against")
    ap.add_argument("--uniform-sub", type=int, default=20000,
                    help="Subsample uniform validation set to this many transitions")
    ap.add_argument("--cap", type=int, default=0, help="Cap validation sets to this many transitions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-mode", choices=["ensemble", "single_member", "both"], default="both",
                    help="Evaluate ensemble average, single member (member 0), or both")
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    harddir = Path(args.hard)

    print("Collecting validation files...", flush=True)
    val_files = {"val_unif": Path(args.uniform)}
    for f in sorted(harddir.glob("val_hard_*.npz")) + sorted(harddir.glob("val_tube_*.npz")):
        if f.exists():
            val_files[f.stem] = f

    # 2. Collect run directories
    run_paths = []
    if args.runs:
        run_paths = [Path(r) for r in args.runs]
    elif args.base_dir:
        base_dir = Path(args.base_dir)
        run_paths = [p for p in base_dir.iterdir() if p.is_dir()]
    else:
        raise ValueError("Either --runs or --base-dir must be provided.")

    rx = re.compile(r"(.+)_seed(\d+)$")
    # variant -> seed -> setname -> dict of metrics
    metrics_by_run: dict = defaultdict(lambda: defaultdict(dict))

    # 3. Evaluate models dataset by dataset (batched by models to avoid I/O bottlenecks)
    print(f"Evaluating models on {device} (eval_mode={args.eval_mode})...", flush=True)
    model_batch_size = 10
    for i in range(0, len(run_paths), model_batch_size):
        run_batch = run_paths[i:i + model_batch_size]
        print(f"--- Loading models batch {i // model_batch_size + 1}/{(len(run_paths) + model_batch_size - 1) // model_batch_size} ---", flush=True)
        
        models = []
        for run in run_batch:
            m = rx.match(run.name)
            if not m:
                models.append((None, None, None))
                continue
            variant, seed = m.group(1), int(m.group(2))
            cfgp, ckptp = run / "config.resolved.json", run / "checkpoint_latest.pt"
            if not (cfgp.exists() and ckptp.exists()):
                models.append((None, None, None))
                continue

            cfg = json.loads(cfgp.read_text())
            model = build_from_cfg(cfg, device)
            try:
                ck = torch.load(ckptp, map_location=device, weights_only=False, mmap=True)
            except TypeError:
                ck = torch.load(ckptp, map_location=device, weights_only=False)

            model.load_state_dict(ck["surrogate"])
            models.append((model, variant, seed))

        for name, path in val_files.items():
            st, nx, pr = load_transitions(path)
            
            # Apply subsampling or capping if specified
            if name == "val_unif" and args.uniform_sub and len(st) > args.uniform_sub:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(st), args.uniform_sub, replace=False)
                st, nx, pr = st[idx], nx[idx], pr[idx]
            elif args.cap and len(st) > args.cap:
                rng = np.random.default_rng(1)
                idx = rng.choice(len(st), args.cap, replace=False)
                st, nx, pr = st[idx], nx[idx], pr[idx]
                
            print(f"  Evaluating on {name} ({len(st)} transitions)", flush=True)

            for model, variant, seed in models:
                if model is None:
                    continue
                
                eval_targets = []
                if args.eval_mode in ("ensemble", "both"):
                    eval_targets.append((model, variant))
                if args.eval_mode in ("single_member", "both") and model.n_models > 1:
                    # Create single-member wrapper for fair single model comparison
                    single_m = EnsembleSurrogate([model.models[0]])
                    eval_targets.append((single_m, f"{variant}__single_member"))

                for m_eval, v_name in eval_targets:
                    rmse, nrmse = calculate_errors(m_eval, st, nx, pr, device)
                    stats = {
                        "rmse_mean": float(rmse.mean()),
                        "rmse_std": float(rmse.std()),
                        "rmse_min": float(rmse.min()),
                        "rmse_max": float(rmse.max()),
                        "rmse_p10": float(np.quantile(rmse, 0.1)),
                        "rmse_p25": float(np.quantile(rmse, 0.25)),
                        "rmse_p50": float(np.quantile(rmse, 0.5)),
                        "rmse_p75": float(np.quantile(rmse, 0.75)),
                        "rmse_p90": float(np.quantile(rmse, 0.9)),
                        "rmse_p95": float(np.quantile(rmse, 0.95)),
                        "rmse_p99": float(np.quantile(rmse, 0.99)),

                        "nrmse_mean": float(nrmse.mean()),
                        "nrmse_std": float(nrmse.std()),
                        "nrmse_min": float(nrmse.min()),
                        "nrmse_max": float(nrmse.max()),
                        "nrmse_p10": float(np.quantile(nrmse, 0.1)),
                        "nrmse_p25": float(np.quantile(nrmse, 0.25)),
                        "nrmse_p50": float(np.quantile(nrmse, 0.5)),
                        "nrmse_p75": float(np.quantile(nrmse, 0.75)),
                        "nrmse_p90": float(np.quantile(nrmse, 0.9)),
                        "nrmse_p95": float(np.quantile(nrmse, 0.95)),
                        "nrmse_p99": float(np.quantile(nrmse, 0.99)),
                    }
                    metrics_by_run[v_name][seed][name] = stats
            del st, nx, pr
            
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Statistical Analysis vs Baselines
    baselines = args.baselines.split(",")
    setnames = sorted(val_files.keys())
    variants = sorted(metrics_by_run.keys())
    
    # We want to analyze each combination of:
    # Set -> Metric -> Variant vs Baseline
    report: dict = {"variants": variants, "sets": setnames, "comparisons": []}

    metric_names = [
        "rmse_mean", "rmse_std", "rmse_min", "rmse_max",
        "rmse_p10", "rmse_p25", "rmse_p50", "rmse_p75", "rmse_p90", "rmse_p95", "rmse_p99",
        "nrmse_mean", "nrmse_std", "nrmse_min", "nrmse_max",
        "nrmse_p10", "nrmse_p25", "nrmse_p50", "nrmse_p75", "nrmse_p90", "nrmse_p95", "nrmse_p99"
    ]

    print("\nStarting statistical comparisons...", flush=True)
    for bl in baselines:
        if bl not in metrics_by_run:
            print(f"Baseline '{bl}' not found in runs, skipping comparison.")
            continue

        print(f"\n==================== COMPARISONS VS BASELINE: {bl} ====================\n")
        # Let's print summary tables for a few key metrics
        for metric in ["rmse_mean", "nrmse_mean", "nrmse_p50", "nrmse_p99"]:
            print(f"--- Metric: {metric} (% change of mean, Welch p-val, MWU p-val) ---")
            hdr = f"{'variant':<22}" + "".join(f"{s[:15]:>16}" for s in setnames)
            print(hdr)
            print("-" * len(hdr))
            
            for v in variants:
                if v == bl:
                    continue
                row = f"{v:<22}"
                for s in setnames:
                    # Get values across seeds
                    ref_vals = np.array([metrics_by_run[bl][seed][s][metric] for seed in metrics_by_run[bl]])
                    v_vals = np.array([metrics_by_run[v][seed][s][metric] for seed in metrics_by_run[v] if s in metrics_by_run[v][seed]])
                    
                    if len(ref_vals) < 2 or len(v_vals) < 2:
                        row += f"{'N/A':>16}"
                        continue
                    
                    pct = 100.0 * (v_vals.mean() - ref_vals.mean()) / (abs(ref_vals.mean()) + 1e-12)
                    tw = ttest_ind(v_vals, ref_vals, equal_var=False)
                    mw = mannwhitneyu(v_vals, ref_vals, alternative="two-sided")
                    
                    # Store detailed record
                    report["comparisons"].append({
                        "baseline": bl,
                        "variant": v,
                        "set": s,
                        "metric": metric,
                        "ref_mean": float(ref_vals.mean()),
                        "ref_std": float(ref_vals.std(ddof=1)),
                        "variant_mean": float(v_vals.mean()),
                        "variant_std": float(v_vals.std(ddof=1)),
                        "pct_change": float(pct),
                        "welch_p": float(tw.pvalue),
                        "mwu_p": float(mw.pvalue)
                    })
                    
                    row += f"{pct:>+6.1f}% (p={tw.pvalue:.2f})"
                print(row)
            print()

        # Compute full statistics for report output
        for metric in metric_names:
            for s in setnames:
                ref_vals = np.array([metrics_by_run[bl][seed][s][metric] for seed in metrics_by_run[bl]])
                for v in variants:
                    if v == bl:
                        continue
                    v_vals = np.array([metrics_by_run[v][seed][s][metric] for seed in metrics_by_run[v] if s in metrics_by_run[v][seed]])
                    if len(ref_vals) >= 2 and len(v_vals) >= 2:
                        pct = 100.0 * (v_vals.mean() - ref_vals.mean()) / (abs(ref_vals.mean()) + 1e-12)
                        tw = ttest_ind(v_vals, ref_vals, equal_var=False)
                        mw = mannwhitneyu(v_vals, ref_vals, alternative="two-sided")
                        report["comparisons"].append({
                            "baseline": bl,
                            "variant": v,
                            "set": s,
                            "metric": metric,
                            "ref_mean": float(ref_vals.mean()),
                            "ref_std": float(ref_vals.std(ddof=1)),
                            "variant_mean": float(v_vals.mean()),
                            "variant_std": float(v_vals.std(ddof=1)),
                            "pct_change": float(pct),
                            "welch_p": float(tw.pvalue),
                            "mwu_p": float(mw.pvalue)
                        })

    out_path = args.out or (Path(args.base_dir) / "campaign_detailed_posthoc_analysis.json" if args.base_dir else Path("campaign_detailed_posthoc_analysis.json"))
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote comprehensive analysis report to {out_path}")


if __name__ == "__main__":
    main()
