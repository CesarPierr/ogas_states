#!/usr/bin/env python3
"""
Trajectory evolution analysis script.
Parses round-by-round metrics across all seeds for all active learning variants
and outputs Markdown tables showing convergence and learning dynamics per round.
"""

import json
import glob
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

METRICS_DEF = [
    # 1. Uniform Attractor
    ("1. Uniform Attractor (RMSE)", "val/rmse_mean"),
    ("1. Uniform Attractor (NRMSE)", "val/nrmse_mean"),
    
    # 2. Hard 1-Step Sets
    ("2. TV-Hard Chocs (RMSE)", "hard_val/val_hard_tv_div/rmse_mean"),
    ("2. Low-Amplitude Calme (RMSE)", "hard_val/val_hard_lowamp_div/rmse_mean"),
    ("2. Mixed Diversity Hard (RMSE)", "hard_val/val_hard_mixed_div/rmse_mean"),
    
    # 3. Off-Attractor Tubes with varying rho & k
    ("3. Tube Low-k rho=0.1 (RMSE)", "hard_val/val_tube_lowk_rho0p1/rmse_mean"),
    ("3. Tube Low-k rho=0.5 (RMSE)", "hard_val/val_tube_lowk_rho0p5/rmse_mean"),
    ("3. Tube Mid-k rho=0.1 (RMSE)", "hard_val/val_tube_midk_rho0p1/rmse_mean"),
    ("3. Tube Mid-k rho=0.25 (RMSE)", "hard_val/val_tube_midk_rho0p25/rmse_mean"),
    ("3. Tube Mid-k rho=0.5 (RMSE)", "hard_val/val_tube_midk_rho0p5/rmse_mean"),
    ("3. Tube High-k rho=0.1 (RMSE)", "hard_val/val_tube_highk_rho0p1/rmse_mean"),
    ("3. Tube High-k rho=0.5 (RMSE)", "hard_val/val_tube_highk_rho0p5/rmse_mean"),
    
    # 4. Long-Horizon Rollout (100 steps)
    ("4. Rollout 100 pas - Mean RMSE", "rollout/rmse_mean"),
    ("4. Rollout 100 pas - Final RMSE (t=100)", "rollout/rmse_final"),
    ("4. Rollout 100 pas - Mean NRMSE", "rollout/nrmse_mean"),
    ("4. Rollout 100 pas - Final NRMSE (t=100)", "rollout/nrmse_final"),
    
    # 5. Generator Diagnostics
    ("5. Generator Rank Corr Difficulte", "generator_metrics/rank_corr"),
    ("5. Generator TV Realism Ratio", "generator_metrics/tv_ratio"),
]

def parse_run_out(file_path):
    rounds = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and '"round"' in line:
                try:
                    d = json.loads(line)
                    r = d.get("round")
                    if r is not None:
                        rounds[int(r)] = d
                except Exception:
                    pass
    return rounds

def collect_campaign_data(base_dirs):
    # variant -> metric -> round -> list of values across seeds
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    variant_seeds = defaultdict(set)
    
    for bdir in base_dirs:
        for p in sorted(glob.glob(f"{bdir}/*/*.out")):
            run_name = Path(p).parent.name
            m = re.match(r"^(.*)_seed(\d+)$", run_name)
            if not m:
                continue
            variant, seed = m.group(1), int(m.group(2))
            variant_seeds[variant].add(seed)
            
            rounds_data = parse_run_out(p)
            for r, metrics in rounds_data.items():
                for label, mkey in METRICS_DEF:
                    if mkey in metrics:
                        val = metrics[mkey]
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            data[variant][label][r].append(float(val))
                            
    return data, variant_seeds

def get_m_size(variant):
    if variant.startswith("ens1_") or "scratch_loss" in variant:
        return 1
    elif variant.startswith("ens3_") or "scratch_ensvar" in variant:
        return 3
    elif variant.startswith("ens5_"):
        return 5
    else:
        return 2  # V3 pilot base default

def format_evolution_markdown(data, variant_seeds, max_round=10):
    lines = ["# Évolution Round-par-Round des Métriques d'Apprentissage Actif\n"]
    lines.append("Ce document détaille la dynamique d'apprentissage et la vitesse de convergence par round.\n")
    lines.append("> **Note de rigueur méthodologique :** Les comparaisons doivent être effectuées **exclusivement à taille d'ensemble M égale**.\n")
    
    all_variants = sorted(data.keys(), key=lambda v: (get_m_size(v), v))
    if not all_variants:
        return "Aucune donnée trouvée."
        
    for label, mkey in METRICS_DEF:
        lines.append(f"## {label}")
        lines.append("")
        
        rounds_seen = set()
        for v in all_variants:
            rounds_seen.update(data[v][label].keys())
        sorted_rounds = sorted(list(rounds_seen))
        if not sorted_rounds:
            continue
            
        header = "| Variante | M | Seeds | " + " | ".join([f"R{r}" for r in sorted_rounds]) + " |"
        sep = "|:---|:---:|:---:|" + "|".join([":---:" for _ in sorted_rounds]) + "|"
        lines.append(header)
        lines.append(sep)
        
        for v in all_variants:
            m_size = get_m_size(v)
            n_seeds = len(variant_seeds[v])
            row = [f"**{v}**", f"M={m_size}", str(n_seeds)]
            for r in sorted_rounds:
                vals = data[v][label].get(r, [])
                if vals:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals) if len(vals) > 1 else 0.0
                    if mean_val < 0.01:
                        row.append(f"{mean_val:.4f} ± {std_val:.4f}")
                    else:
                        row.append(f"{mean_val:.3f} ± {std_val:.3f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("\n---\n")
        
    return "\n".join(lines)

def main():
    base_dirs = [
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_pure_scratch",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_ensemble_scaling",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_v3_pilot",
    ]
    
    data, variant_seeds = collect_campaign_data(base_dirs)
    md = format_evolution_markdown(data, variant_seeds)
    
    out_file = Path("docs/round_evolution.md")
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text(md)
    print(f"Rapport d'évolution round-par-round écrit dans {out_file}")
    
    # Save structured JSON
    clean_data = {}
    for v in data:
        clean_data[v] = {
            "n_seeds": len(variant_seeds[v]),
            "seeds": sorted(list(variant_seeds[v])),
            "metrics": {}
        }
        for label in data[v]:
            clean_data[v]["metrics"][label] = {
                r: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
                for r, vals in data[v][label].items()
            }
    
    json_out = Path("round_evolution_analysis.json")
    with open(json_out, "w") as f:
        json.dump(clean_data, f, indent=2)
    print(f"Données structurées sauvegardées dans {json_out}")

if __name__ == "__main__":
    main()
