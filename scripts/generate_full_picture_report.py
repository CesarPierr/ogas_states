#!/usr/bin/env python3
"""
Full Picture Exhaustive Report Generator.
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
    
    ("4. Ensembles Difficiles & Dérives Physiques", "TV-Hard Chocs - RMSE Mean", "hard_val/val_hard_tv_div/rmse_mean", "RMSE"),
    ("4. Ensembles Difficiles & Dérives Physiques", "Low-Amplitude Calme - RMSE Mean", "hard_val/val_hard_lowamp_div/rmse_mean", "RMSE"),
    ("4. Ensembles Difficiles & Dérives Physiques", "Mixed Diversity Hard - RMSE Mean", "hard_val/val_hard_mixed_div/rmse_mean", "RMSE"),
    
    ("5. Rollout Autorégressif Long-Horizon (t=100)", "Rollout 100 pas - Final RMSE (t=100)", "rollout/rmse_final", "Rollout RMSE"),
    ("5. Rollout Autorégressif Long-Horizon (t=100)", "Rollout 100 pas - Mean Horizon RMSE", "rollout/rmse_mean", "Rollout RMSE"),
    ("5. Rollout Autorégressif Long-Horizon (t=100)", "Rollout 100 pas - Final NRMSE (t=100)", "rollout/nrmse_final", "Rollout NRMSE"),
]

def load_all_runs(base_dirs):
    # runs[variant][seed] = list of round dicts
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
            # Fallback to .out parsing
            out_files = sorted(glob.glob(f"{p}/*.out"))
            if out_files:
                parsed = []
                for out_f in out_files:
                    with open(out_f, "r", errors="ignore") as f:
                        for line in f:
                            if line.startswith("{") and '"round"' in line:
                                try:
                                    parsed.append(json.loads(line))
                                except:
                                    pass
                if parsed:
                    runs[vname][seed] = parsed
    return runs

def format_num(val):
    if np.isnan(val):
        return "-"
    if abs(val) < 0.001:
        return f"{val:.5f}"
    elif abs(val) < 0.01:
        return f"{val:.4f}"
    elif abs(val) < 10.0:
        return f"{val:.3f}"
    elif abs(val) < 100.0:
        return f"{val:.2f}"
    else:
        return f"{val:.1f}"

def build_report():
    base_dirs = [
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_pure_scratch",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_ensemble_scaling",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_v3_pilot",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_heuristic_tube",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_m1_improvements",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_classical_al",
    ]
    
    runs = load_all_runs(base_dirs)
    
    lines = [
        "# Rapport Exhaustif de Synthèse Multi-Dimensionnelle & Statistique Robuste",
        "",
        "Ce rapport consolide l'ensemble des résultats de la campagne d'apprentissage actif pour les régimes $M=1$, $M=3$ et $M=5$.",
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
    
    # Section 1: Overview Figures
    lines.extend([
        "## 1. Visualisations Synthétiques & Trade-offs",
        "",
        "![Figure 10 : Robustesse Statistique des Rollouts (Médiane vs Moyenne & IQR 25-75)](figures/fig10_robust_median_rollout_comparison.png)",
        "",
        "![Figure 1: In-Distribution vs Out-of-Distribution Tube Error Tradeoff](figures/fig1_tradeoff_tube_vs_indist.png)",
        "",
        "![Figure 9: Rollout Horizon Error Growth (Log Scale)](figures/fig9_rollout_horizon_time_series.png)",
        "",
        "---",
        ""
    ])
    
    # Section 2: Comprehensive Multi-Statistical Tables at Round 9 (or latest available)
    lines.append("## 2. Tableaux Statistiques Exhaustifs par Régime d'Ensemble (Round 9 / Final)\n")
    
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
                for s, hist in seeds_dict.items():
                    # Pick round 9 or latest available round
                    entry = next((item for item in hist if item.get("round") == 9), None)
                    if entry is None and hist:
                        entry = hist[-1]
                    if entry and mkey in entry:
                        x = entry[mkey]
                        if isinstance(x, (int, float)) and not np.isnan(x):
                            vals.append(float(x))
                            
                if not vals:
                    continue
                    
                arr = np.array(vals)
                n_s = len(arr)
                mean_val = np.mean(arr)
                std_val = np.std(arr) if n_s > 1 else 0.0
                med_val = np.median(arr)
                p25_val = np.percentile(arr, 25)
                p75_val = np.percentile(arr, 75)
                p90_val = np.percentile(arr, 90)
                p95_val = np.percentile(arr, 95)
                p99_val = np.percentile(arr, 99)
                min_val = np.min(arr)
                max_val = np.max(arr)
                
                row = [
                    f"**{v}**",
                    f"{n_s}/10",
                    f"{format_num(mean_val)} ± {format_num(std_val)}",
                    f"**{format_num(med_val)}**",
                    f"[{format_num(p25_val)}, {format_num(p75_val)}]",
                    format_num(p90_val),
                    format_num(p95_val),
                    format_num(p99_val),
                    format_num(min_val),
                    format_num(max_val)
                ]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
            
    # Section 3: Round by Round Trajectories
    lines.append("## 3. Dynamique Temporelle par Round (R0 à R9) : Moyenne ± Std [Médiane]\n")
    
    core_metrics = [
        ("val/rmse_mean", "Attracteur Uniforme - RMSE Mean"),
        ("hard_val/val_tube_midk_rho0p5/rmse_mean", "Tube Mid-k (rho=0.5) - RMSE Mean"),
        ("rollout/rmse_final", "Rollout Final RMSE (t=100)"),
    ]
    
    for m_target in [1, 3, 5]:
        vars_in_m = [v for v in sorted(runs.keys()) if get_m_size(v) == m_target]
        if not vars_in_m:
            continue
            
        lines.append(f"### Trajectoires R0-R9 pour Régime M={m_target}\n")
        
        for mkey, mtitle in core_metrics:
            lines.append(f"#### {mtitle} (M={m_target})\n")
            
            # Determine available rounds
            all_rounds = set()
            for v in vars_in_m:
                for s, hist in runs[v].items():
                    for item in hist:
                        if mkey in item and "round" in item:
                            all_rounds.add(int(item["round"]))
                            
            sorted_rounds = sorted(list(all_rounds))
            if not sorted_rounds:
                continue
                
            header = "| Variante | " + " | ".join([f"R{r}" for r in sorted_rounds]) + " |"
            sep = "|:---|" + "|".join([":---:" for _ in sorted_rounds]) + "|"
            lines.append(header)
            lines.append(sep)
            
            for v in vars_in_m:
                row = [f"**{v}**"]
                for r in sorted_rounds:
                    vals_r = []
                    for s, hist in runs[v].items():
                        entry = next((item for item in hist if item.get("round") == r), None)
                        if entry and mkey in entry:
                            x = entry[mkey]
                            if isinstance(x, (int, float)) and not np.isnan(x):
                                vals_r.append(float(x))
                    if vals_r:
                        arr = np.array(vals_r)
                        mn, std, med = np.mean(arr), np.std(arr), np.median(arr)
                        row.append(f"{format_num(mn)} ± {format_num(std)} *(med: {format_num(med)})*")
                    else:
                        row.append("-")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
            
    out_path = Path("docs/full_picture_report.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated comprehensive report at {out_path} ({len(lines)} lines)")

if __name__ == "__main__":
    build_report()
