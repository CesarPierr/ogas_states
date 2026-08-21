#!/usr/bin/env python3
"""
Comprehensive Statistical Matrix Exporter for 1D Burgers Profile B (10 Seeds)
=============================================================================
Computes full distribution statistics across all 10 seeds for all 5 strategies:
- Mean +/- Std
- Median (q50) & IQR (q75 - q25)
- Tail Quantiles (p90, p95, p99)
- 1-Step NRMSE, 1-Step RMSE
- Rollout NRMSE Mean, Rollout RMSE Final (T=40), Rollout NRMSE p95/p99

Generates formatted Markdown & CSV tables.
"""
import glob
import json
from pathlib import Path
import numpy as np


BASE_DIR = Path("/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/burgers_profile_b_sweep")
STRATEGIES = [
    ("uniform_baseline", "Uniform Baseline (On-Attractor)"),
    ("heuristic_tube", "Heuristic Tube Perturbations"),
    ("classic_al_topk", "Classical AL (Top-K Uncertainty)"),
    ("classic_al_sbal", "Classical AL (SBAL Sampling)"),
    ("ogas_generative", "OGAS (Generative Active Sampling)"),
]


def extract_all_metrics():
    results = {}
    
    for key, label in STRATEGIES:
        strat_runs = sorted(glob.glob(f"{BASE_DIR}/{key}_*"))
        strat_metrics = {
            "val_nrmse_mean": [],
            "val_rmse_mean": [],
            "val_nrmse_p50": [],
            "val_nrmse_p90": [],
            "val_nrmse_p95": [],
            "val_nrmse_p99": [],
            "val_rmse_p95": [],
            "val_rmse_p99": [],
            "rollout_nrmse_mean": [],
            "rollout_rmse_mean": [],
            "rollout_nrmse_final": [],
            "rollout_rmse_final": [],
            "rollout_nrmse_mean_p95": [],
            "rollout_nrmse_mean_p99": [],
            "rollout_rmse_mean_p95": [],
            "rollout_rmse_mean_p99": [],
        }
        
        for r in strat_runs:
            h_file = Path(r) / "history.json"
            if h_file.exists():
                try:
                    with open(h_file) as f:
                        h = json.load(f)
                        if len(h) >= 10:
                            last = h[-1]
                            strat_metrics["val_nrmse_mean"].append(last.get("val/nrmse_mean", np.nan))
                            strat_metrics["val_rmse_mean"].append(last.get("val/rmse_mean", np.nan))
                            strat_metrics["val_nrmse_p50"].append(last.get("val/nrmse_p50", np.nan))
                            strat_metrics["val_nrmse_p90"].append(last.get("val/nrmse_p90", np.nan))
                            strat_metrics["val_nrmse_p95"].append(last.get("val/nrmse_p95", np.nan))
                            strat_metrics["val_nrmse_p99"].append(last.get("val/nrmse_p99", np.nan))
                            strat_metrics["val_rmse_p95"].append(last.get("val/rmse_p95", np.nan))
                            strat_metrics["val_rmse_p99"].append(last.get("val/rmse_p99", np.nan))
                            strat_metrics["rollout_nrmse_mean"].append(last.get("rollout/nrmse_mean", np.nan))
                            strat_metrics["rollout_rmse_mean"].append(last.get("rollout/rmse_mean", np.nan))
                            strat_metrics["rollout_nrmse_final"].append(last.get("rollout/nrmse_final", np.nan))
                            strat_metrics["rollout_rmse_final"].append(last.get("rollout/rmse_final", np.nan))
                            strat_metrics["rollout_nrmse_mean_p95"].append(last.get("rollout/nrmse_mean_p95", np.nan))
                            strat_metrics["rollout_nrmse_mean_p99"].append(last.get("rollout/nrmse_mean_p99", np.nan))
                            strat_metrics["rollout_rmse_mean_p95"].append(last.get("rollout/rmse_mean_p95", np.nan))
                            strat_metrics["rollout_rmse_mean_p99"].append(last.get("rollout/rmse_mean_p99", np.nan))
                except Exception:
                    pass
                    
        results[key] = {
            "label": label,
            "completed": len(strat_metrics["rollout_nrmse_mean"]),
            "data": strat_metrics,
        }
        
    return results


def format_stat(arr):
    arr = np.array([x for x in arr if not np.isnan(x)])
    if len(arr) == 0:
        return "N/A"
    mean = np.mean(arr)
    std = np.std(arr)
    median = np.median(arr)
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    return f"{mean:.4f} +/- {std:.4f} (med: {median:.4f}, iqr: {iqr:.4f})"


def format_mean_std(arr):
    arr = np.array([x for x in arr if not np.isnan(x)])
    if len(arr) == 0:
        return "N/A"
    return f"{np.mean(arr):.4f} +/- {np.std(arr):.4f}"


def format_median_iqr(arr):
    arr = np.array([x for x in arr if not np.isnan(x)])
    if len(arr) == 0:
        return "N/A"
    med = np.median(arr)
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    return f"{med:.4f} [{iqr:.4f}]"


def print_matrices(results):
    print("=" * 125)
    print("1. 1D BURGERS PROFILE B (Re in [2000, 10000]) - AUTOREGRESSIVE ROLLOUT (T=40) STATISTICAL MATRIX (10 SEEDS)")
    print("=" * 125)
    print(f"{'Strategy':32s} | {'Completed':9s} | {'Rollout NRMSE Mean':22s} | {'Rollout Final (T=40)':22s} | {'Rollout p95 Tail':22s}")
    print("-" * 125)
    for key, val in results.items():
        d = val["data"]
        c = f"{val['completed']}/10"
        r_mean = format_mean_std(d["rollout_nrmse_mean"])
        r_fin = format_mean_std(d["rollout_nrmse_final"])
        r_p95 = format_mean_std(d["rollout_nrmse_mean_p95"])
        print(f"{val['label']:32s} | {c:9s} | {r_mean:22s} | {r_fin:22s} | {r_p95:22s}")
        
    print("\n" + "=" * 125)
    print("2. 1D BURGERS PROFILE B - 1-STEP PREDICTION & TAIL SHOCK ERRORS (10 SEEDS)")
    print("=" * 125)
    print(f"{'Strategy':32s} | {'1-Step NRMSE Mean':22s} | {'1-Step Median [IQR]':22s} | {'1-Step NRMSE p95':22s} | {'1-Step NRMSE p99':22s}")
    print("-" * 125)
    for key, val in results.items():
        d = val["data"]
        s_mean = format_mean_std(d["val_nrmse_mean"])
        s_med = format_median_iqr(d["val_nrmse_mean"])
        s_p95 = format_mean_std(d["val_nrmse_p95"])
        s_p99 = format_mean_std(d["val_nrmse_p99"])
        print(f"{val['label']:32s} | {s_mean:22s} | {s_med:22s} | {s_p95:22s} | {s_p99:22s}")
    print("=" * 125)


def export_markdown_report(results, out_path: Path):
    lines = [
        "# Comprehensive Statistical Benchmark Matrix: 1D Burgers Profile B (10 Seeds)",
        "",
        "**Regime:** High-Reynolds ($Re \\in [2000, 10000]$, $\\nu \\in [10^{-4}, 5 \\times 10^{-4}]$), Razor-sharp persistent shocks ($T=40$ steps rollout).",
        "",
        "## 1. Multi-Step Autoregressive Rollout Benchmark ($T=40$ Horizons)",
        "",
        "| Strategy | Completed | Rollout NRMSE Mean | Rollout NRMSE Median [IQR] | Rollout Final Step ($T=40$) | Rollout Tail Error (p95) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    
    for key, val in results.items():
        d = val["data"]
        lines.append(
            f"| **{val['label']}** | {val['completed']}/10 | "
            f"`{format_mean_std(d['rollout_nrmse_mean'])}` | `{format_median_iqr(d['rollout_nrmse_mean'])}` | "
            f"`{format_mean_std(d['rollout_nrmse_final'])}` | `{format_mean_std(d['rollout_nrmse_mean_p95'])}` |"
        )
        
    lines.extend([
        "",
        "## 2. 1-Step Prediction & Out-of-Distribution Tail Errors",
        "",
        "| Strategy | 1-Step NRMSE Mean | 1-Step NRMSE Median [IQR] | 1-Step Tail p90 | 1-Step Tail p95 | 1-Step Tail p99 (Worst Case) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    
    for key, val in results.items():
        d = val["data"]
        lines.append(
            f"| **{val['label']}** | `{format_mean_std(d['val_nrmse_mean'])}` | `{format_median_iqr(d['val_nrmse_mean'])}` | "
            f"`{format_mean_std(d['val_nrmse_p90'])}` | `{format_mean_std(d['val_nrmse_p95'])}` | `{format_mean_std(d['val_nrmse_p99'])}` |"
        )
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\n[REPORT EXPORTED] Saved comprehensive matrix to {out_path}")


def main():
    results = extract_all_metrics()
    print_matrices(results)
    export_markdown_report(results, Path("docs/burgers_profile_b_statistical_matrix.md"))


if __name__ == "__main__":
    main()
