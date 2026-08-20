#!/usr/bin/env python3
"""
Burgers 1D Full Picture Exhaustive Report Generator.
Generates comprehensive multi-dimensional statistical tables across:
- Models (M = 1, 3, 5)
- Active Learning Methods vs Controls vs Heuristic Tube
- All Evaluation Benchmarks (Uniform, Hard suites, Tubes at varying rho, Rollouts)
- Complete Statistical Aggregations across seeds:
  * Mean +/- Std
  * Median (p50)
  * Interquartile Range IQR [p25, p75]
  * Extreme Quantiles: p90, p95, p99
  * Range: Min, Max
- Trajectory evolution across rounds (R0 to R9).
"""

import json
import glob
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

def get_m_size(variant):
    if variant.startswith("ens1_") or "scratch_loss" in variant:
        return 1
    elif variant.startswith("ens3_") or "scratch_ensvar" in variant:
        return 3
    elif variant.startswith("ens5_"):
        return 5
    else:
        return 2

KEY_BENCHMARKS = [
    # Category, Label, Metric Key, Unit
    ("1. Attracteur In-Distribution (1-pas)", "Attracteur - RMSE Mean", "val/rmse_mean", "RMSE"),
    ("1. Attracteur In-Distribution (1-pas)", "Attracteur - NRMSE Mean", "val/nrmse_mean", "NRMSE"),
    
    ("2. Robustesse Tube Hors-Attracteur (rho=0.5)", "Tube Low-k (rho=0.5) - RMSE", "hard_val/val_tube_lowk_rho0p5/rmse_mean", "RMSE"),
    ("2. Robustesse Tube Hors-Attracteur (rho=0.5)", "Tube Mid-k (rho=0.5) - RMSE", "hard_val/val_tube_midk_rho0p5/rmse_mean", "RMSE"),
    ("2. Robustesse Tube Hors-Attracteur (rho=0.5)", "Tube High-k (rho=0.5) - RMSE", "hard_val/val_tube_highk_rho0p5/rmse_mean", "RMSE"),
    
    ("3. Robustesse Tube Hors-Attracteur (rho=0.1)", "Tube Low-k (rho=0.1) - RMSE", "hard_val/val_tube_lowk_rho0p1/rmse_mean", "RMSE"),
    ("3. Robustesse Tube Hors-Attracteur (rho=0.1)", "Tube Mid-k (rho=0.1) - RMSE", "hard_val/val_tube_midk_rho0p1/rmse_mean", "RMSE"),
    ("3. Robustesse Tube Hors-Attracteur (rho=0.1)", "Tube High-k (rho=0.1) - RMSE", "hard_val/val_tube_highk_rho0p1/rmse_mean", "RMSE"),
    
    ("4. Ensembles Difficiles & Chocs Raides", "TV-Hard Chocs - RMSE Mean", "hard_val/val_hard_tv_div/rmse_mean", "RMSE"),
    ("4. Ensembles Difficiles & Chocs Raides", "Low-Amplitude Calme - RMSE Mean", "hard_val/val_hard_lowamp_div/rmse_mean", "RMSE"),
    ("4. Ensembles Difficiles & Chocs Raides", "Mixed Diversity Hard - RMSE Mean", "hard_val/val_hard_mixed_div/rmse_mean", "RMSE"),
    
    ("5. Rollout Autorégressif Long-Horizon", "Rollout - Final RMSE", "rollout/rmse_final", "Rollout RMSE"),
    ("5. Rollout Autorégressif Long-Horizon", "Rollout - Mean Horizon RMSE", "rollout/rmse_mean", "Rollout RMSE"),
    ("5. Rollout Autorégressif Long-Horizon", "Rollout - Final NRMSE", "rollout/nrmse_final", "Rollout NRMSE"),
]

def load_all_runs(base_dirs):
    runs = defaultdict(dict)
    for bdir in base_dirs:
        for p in sorted(glob.glob(f"{bdir}/*")):
            p = Path(p)
            if not p.is_dir():
                continue
            rname = p.name
            m = re.match(r"^(.*)_seed(\d+)$", rname)
            if not m:
                continue
            vname, seed = m.group(1), int(m.group(2))
            
            hist_file = p / "history.json"
            if hist_file.exists():
                try:
                    h = json.loads(hist_file.read_text())
                    if isinstance(h, list) and h:
                        runs[vname][seed] = h
                        continue
                except:
                    pass
    return runs

def format_num(val):
    if np.isnan(val):
        return "-"
    if abs(val) < 0.0001:
        return f"{val:.6f}"
    elif abs(val) < 0.001:
        return f"{val:.5f}"
    elif abs(val) < 0.01:
        return f"{val:.4f}"
    elif abs(val) < 10.0:
        return f"{val:.3f}"
    elif abs(val) < 100.0:
        return f"{val:.2f}"
    else:
        return f"{val:.1f}"

def get_nested(d, key):
    if key in d:
        v = d[key]
        return float(v) if isinstance(v, (int, float)) else np.nan
    parts = key.split("/")
    curr = d
    for p in parts:
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
        else:
            return np.nan
    return float(curr) if isinstance(curr, (int, float)) else np.nan

def build_report():
    base_dirs = [
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/burgers_benchmark",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/burgers_hard2",
    ]
    
    runs = load_all_runs(base_dirs)
    
    lines = [
        "# Rapport Exhaustif de Synthèse Multi-Dimensionnelle — Burgers 1D",
        "",
        "Ce rapport consolide l'ensemble des résultats de la campagne d'apprentissage actif sur le système de Burgers 1D ($N=256$, formation de chocs) pour les régimes $M=1$, $M=3$ et $M=5$ (10 graines par variante, 10 rounds iso-budget).",
        "",
        "> [!IMPORTANT]",
        "> **Méthodologie Statistique Exhaustive :**",
        "> Pour chaque métrique, nous rapportons la distribution complète à travers les 10 graines indépendantes :",
        "> - **Moyenne $\pm$ Écart-type (Std)**",
        "> - **Médiane (p50)** et **Intervalle Interquartile IQR [p25, p75]**",
        "> - **Quantiles de queue : p90, p95, p99**",
        "> - **Extrêmes : Min (Meilleur run) et Max (Pire run / Dérive)**",
        "",
        "---",
        ""
    ]
    
    # Section: Comprehensive Multi-Statistical Tables at Round 9 (or latest available)
    lines.append("## Tableaux Statistiques Exhaustifs par Régime d'Ensemble (Round 9 / Final)\n")
    
    for m_target in [1, 3, 5]:
        vars_in_m = [v for v in sorted(runs.keys()) if get_m_size(v) == m_target]
        if not vars_in_m:
            continue
            
        lines.append(f"# ==========================================================================")
        lines.append(f"# RÉGIME M={m_target} (Taille du Comité = {m_target})")
        lines.append(f"# ==========================================================================\n")
        
        current_cat = ""
        for cat_name, label, mkey, unit in KEY_BENCHMARKS:
            if cat_name != current_cat:
                current_cat = cat_name
                lines.append(f"### {current_cat}\n")
                
            lines.append(f"#### {label} ({unit})\n")
            
            header = "| Variante | Graines | Moyenne ± Std | **Médiane (p50)** | **IQR [Q25, Q75]** | p90 | p95 | p99 | Min | Max |"
            sep = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
            lines.append(header)
            lines.append(sep)
            
            for v in vars_in_m:
                seeds_dict = runs[v]
                vals = []
                for s, h in seeds_dict.items():
                    if h:
                        last_r = h[-1]
                        val = get_nested(last_r, mkey)
                        if not np.isnan(val):
                            vals.append(val)
                            
                n_seeds = len(vals)
                if n_seeds == 0:
                    continue
                arr = np.array(vals)
                mean, std = np.mean(arr), np.std(arr)
                p25, p50, p75 = np.percentile(arr, [25, 50, 75])
                p90, p95, p99 = np.percentile(arr, [90, 95, 99])
                min_v, max_v = np.min(arr), np.max(arr)
                
                row = (
                    f"| **{v}** | {n_seeds}/10 | "
                    f"{format_num(mean)} ± {format_num(std)} | "
                    f"**{format_num(p50)}** | "
                    f"[{format_num(p25)}, {format_num(p75)}] | "
                    f"{format_num(p90)} | {format_num(p95)} | {format_num(p99)} | "
                    f"{format_num(min_v)} | {format_num(max_v)} |"
                )
                lines.append(row)
            lines.append("")
            
    out_path = Path("docs/burgers_full_picture_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Report written to {out_path} ({len(lines)} lines)")

if __name__ == "__main__":
    build_report()
