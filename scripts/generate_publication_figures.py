#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for Active Learning in Dynamical Systems.
Generates rich figures comparing Active Learning variants against controls at strictly equal M.
All plots include mean +/- std error bands and robust median/IQR computed over 10 independent random seeds.
Includes:
- Control Baselines (Uniform)
- Heuristic Baselines (Random Noise Tube) across M=1, 3, 5
- Active Learning Methods (Loss-Spectrum, EnsVar, Gen-V3)
- Improved M=1 Variants (Sobolev H^1, Jitter Replay, Spectral Edit Smooth)
"""

import os
import re
import glob
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Color palette definition
COLORS = {
    "baseline": "#4A5568",      # Slate Gray (Uniform)
    "random_tube": "#DD6B20",   # Amber / Orange (Random Noise Heuristic)
    "spectrum": "#3182CE",      # Deep Sky Blue (Loss-Spectrum Raw)
    "white": "#E2E8F0",         # Light Gray
    "gen_edit": "#38A169",      # Emerald Green (Gen-V3 / Edit Smooth)
    "ensvar": "#805AD5",        # Violet / Purple (EnsVar 0.5)
    "sobolev": "#6B46C1",       # Deep Purple (Sobolev H1)
    "jitter": "#D53F8C",        # Crimson Magenta (Jitter Replay)
}

def get_m_size(variant):
    if variant.startswith("ens1_") or "scratch_loss" in variant:
        return 1
    elif variant.startswith("ens3_") or "scratch_ensvar" in variant:
        return 3
    elif variant.startswith("ens5_"):
        return 5
    else:
        return 2

def load_all_campaign_data():
    base_dirs = [
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_pure_scratch",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_ensemble_scaling",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_v3_pilot",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_heuristic_tube",
        "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_m1_improvements",
    ]
    
    # Structure: data[variant][metric_key][round] = [val_seed1, val_seed2, ...]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    variant_seeds = defaultdict(set)
    
    for bdir in base_dirs:
        for p in sorted(glob.glob(f"{bdir}/*")):
            p = Path(p)
            if not p.is_dir():
                continue
            rname = p.name
            m = re.match(r"^(.*)_seed(\d+)$", rname)
            if not m:
                continue
            variant, seed = m.group(1), int(m.group(2))
            variant_seeds[variant].add(seed)
            
            rounds_data = []
            hist_file = p / "history.json"
            if hist_file.exists():
                try:
                    h = json.loads(hist_file.read_text())
                    if isinstance(h, list) and h:
                        rounds_data = h
                except:
                    pass
            if not rounds_data:
                out_files = sorted(glob.glob(f"{p}/*.out"))
                for out_f in out_files:
                    with open(out_f, "r", errors="ignore") as f:
                        for line in f:
                            if line.startswith("{") and '"round"' in line:
                                try:
                                    rounds_data.append(json.loads(line))
                                except:
                                    pass
                                    
            for d in rounds_data:
                r = d.get("round")
                if r is None:
                    continue
                for mkey, val in d.items():
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        data[variant][mkey][int(r)].append(float(val))
                        
    return data, variant_seeds

def format_variant_name(v):
    name_map = {
        "ens1_uniform_baseline": "Uniform Control (M=1)",
        "ens1_heuristic_tube": "Random Noise Tube (M=1)",
        "pure_scratch_loss_spectrum": "Loss-Spectrum Raw (M=1)",
        "ens1_loss_edit_smooth": "Spectral Smooth Edit (M=1)",
        "ens1_loss_jitter_replay": "Jitter Replay Buffer (M=1)",
        "ens1_loss_sobolev_tv": "Sobolev H¹ Regularized (M=1)",
        "ens3_uniform_baseline": "Uniform Control (M=3)",
        "ens3_heuristic_tube": "Random Noise Tube (M=3)",
        "pure_scratch_ensvar_white": "EnsVar White Noise (M=3)",
        "ens3_gen_v3_edit": "Gen-V3 Active Learning (M=3)",
        "ens3_ensvar_0p5": "EnsVar 0.5 (M=3)",
        "ens5_uniform_baseline": "Uniform Control (M=5)",
        "ens5_heuristic_tube": "Random Noise Tube (M=5)",
        "ens5_ensvar_0p5": "EnsVar 0.5 (M=5)",
        "ens5_gen_v3_edit": "Gen-V3 Active Learning (M=5)",
    }
    return name_map.get(v, v)

def get_color_for_variant(v):
    if "uniform_baseline" in v:
        return COLORS["baseline"]
    elif "heuristic_tube" in v or "random_tube" in v:
        return COLORS["random_tube"]
    elif "loss_spectrum" in v:
        return COLORS["spectrum"]
    elif "loss_sobolev" in v:
        return COLORS["sobolev"]
    elif "loss_jitter" in v:
        return COLORS["jitter"]
    elif "gen_v3_edit" in v or "loss_edit_smooth" in v:
        return COLORS["gen_edit"]
    elif "ensvar_0p5" in v or "ensvar_white" in v:
        return COLORS["ensvar"]
    else:
        return "#ED8936"

# -----------------------------------------------------------------------------
# Figure 1: Convergence Across Rounds with Error Bands (Iso-M Panels)
# -----------------------------------------------------------------------------
def plot_figure_1_convergence(data, out_dir):
    fig, axes = plt.subplots(3, 3, figsize=(19, 14), sharex="col", dpi=200)
    fig.suptitle("Figure 1 : Convergence Dynamics Across Active Learning Rounds (mean ± std, 10 seeds)", fontsize=16, fontweight="bold", y=0.98)

    benchmarks = [
        ("hard_val/val_tube_midk_rho0p5/rmse_mean", "Out-of-Attractor: Tube Mid-k (ρ=0.5) RMSE", True),
        ("hard_val/val_hard_tv_div/rmse_mean", "High Gradient Shocks: TV-Hard RMSE", False),
        ("val/rmse_mean", "Attractor Nominal: Uniform RMSE", False),
    ]

    m_configs = [
        (1, ["ens1_uniform_baseline", "ens1_heuristic_tube", "pure_scratch_loss_spectrum", "ens1_loss_edit_smooth", "ens1_loss_jitter_replay", "ens1_loss_sobolev_tv"]),
        (3, ["ens3_uniform_baseline", "ens3_heuristic_tube", "ens3_ensvar_0p5", "ens3_gen_v3_edit"]),
        (5, ["ens5_uniform_baseline", "ens5_heuristic_tube", "ens5_ensvar_0p5", "ens5_gen_v3_edit"]),
    ]

    for col_idx, (m_val, var_list) in enumerate(m_configs):
        for row_idx, (mkey, title, is_log) in enumerate(benchmarks):
            ax = axes[row_idx, col_idx]
            ax.grid(True, linestyle="--", alpha=0.4)
            
            for v in var_list:
                if v not in data or mkey not in data[v]:
                    continue
                r_dict = data[v][mkey]
                sorted_r = sorted(r_dict.keys())
                if not sorted_r:
                    continue
                
                means = np.array([np.mean(r_dict[r]) for r in sorted_r])
                stds = np.array([np.std(r_dict[r]) if len(r_dict[r]) > 1 else 0.0 for r in sorted_r])
                color = get_color_for_variant(v)
                label = format_variant_name(v).split(" (")[0]
                ls = "--" if "uniform" in v else ("-." if "heuristic" in v else "-")
                marker = "o" if "uniform" in v else ("^" if "heuristic" in v else "s")
                
                ax.plot(sorted_r, means, label=label, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=5)
                ax.fill_between(sorted_r, np.maximum(0, means - stds), means + stds, color=color, alpha=0.15)
                
            if row_idx == 0:
                ax.set_title(f"Régime M = {m_val}", fontsize=14, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(title, fontsize=11.5, fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel("Active Learning Round", fontsize=11.5, fontweight="bold")
                
            if is_log:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.legend(loc="upper right" if row_idx == 0 else "best", framealpha=0.85, fontsize=8.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = out_dir / "fig1_convergence_iso_m.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved {out_path}")

# -----------------------------------------------------------------------------
# Figure 2: Out-of-Attractor Sensitivity to Perturbation Intensity (rho in [0.1, 0.5])
# -----------------------------------------------------------------------------
def plot_figure_2_rho_sensitivity(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, dpi=200)
    fig.suptitle("Figure 2 : Out-of-Attractor Generalization vs Perturbation Intensity (Mid-k, Final Round)", fontsize=15, fontweight="bold", y=0.98)
    
    rhos = [0.1, 0.25, 0.5]
    rho_keys = [
        "hard_val/val_tube_midk_rho0p1/rmse_mean",
        "hard_val/val_tube_midk_rho0p25/rmse_mean",
        "hard_val/val_tube_midk_rho0p5/rmse_mean",
    ]
    
    m_configs = [
        (1, ["ens1_uniform_baseline", "ens1_heuristic_tube", "pure_scratch_loss_spectrum", "ens1_loss_edit_smooth", "ens1_loss_jitter_replay", "ens1_loss_sobolev_tv"]),
        (3, ["ens3_uniform_baseline", "ens3_heuristic_tube", "ens3_ensvar_0p5", "ens3_gen_v3_edit"]),
        (5, ["ens5_uniform_baseline", "ens5_heuristic_tube", "ens5_ensvar_0p5", "ens5_gen_v3_edit"]),
    ]
    
    for idx, (m_val, var_list) in enumerate(m_configs):
        ax = axes[idx]
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(f"Régime M = {m_val}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Perturbation Radius ρ", fontsize=12, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("RMSE Mean (Final Round)", fontsize=12, fontweight="bold")
            
        for v in var_list:
            if v not in data:
                continue
            means = []
            stds = []
            valid = True
            for rk in rho_keys:
                if rk not in data[v] or not data[v][rk]:
                    valid = False
                    break
                max_r = max(data[v][rk].keys())
                vals = data[v][rk][max_r]
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            if not valid:
                continue
                
            means = np.array(means)
            stds = np.array(stds)
            color = get_color_for_variant(v)
            label = format_variant_name(v).split(" (")[0]
            ls = "--" if "uniform" in v else ("-." if "heuristic" in v else "-")
            marker = "o" if "uniform" in v else ("^" if "heuristic" in v else "s")
            
            ax.plot(rhos, means, label=label, color=color, linestyle=ls, marker=marker, linewidth=2.2)
            ax.fill_between(rhos, np.maximum(0, means - stds), means + stds, color=color, alpha=0.15)
            
        ax.set_xticks(rhos)
        ax.set_xticklabels(["ρ=0.10\n(Faible)", "ρ=0.25\n(Moyen)", "ρ=0.50\n(Fort)"])
        ax.legend(loc="upper left", framealpha=0.85, fontsize=8.5)
        
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path = out_dir / "fig2_tube_perturbation_sensitivity.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved {out_path}")

# -----------------------------------------------------------------------------
# Figure 3: Error Quantile Compression (Tail Distribution p50 -> p99)
# -----------------------------------------------------------------------------
def plot_figure_3_quantiles(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200)
    fig.suptitle("Figure 3 : Error Distribution Tail Compression Across Quantiles (Uniform Attractor, M=1, 3, 5)", fontsize=15, fontweight="bold", y=0.98)
    
    q_keys = [
        ("p50", "val/rmse_p50"),
        ("p75", "val/rmse_p75"),
        ("p90", "val/rmse_p90"),
        ("p95", "val/rmse_p95"),
        ("p99", "val/rmse_p99"),
    ]
    q_labels = ["p50 (Med)", "p75", "p90", "p95", "p99 (Tail)"]
    
    comparisons = [
        ("Régime M=1", ["ens1_uniform_baseline", "ens1_heuristic_tube", "pure_scratch_loss_spectrum", "ens1_loss_edit_smooth", "ens1_loss_jitter_replay", "ens1_loss_sobolev_tv"]),
        ("Régime M=3", ["ens3_uniform_baseline", "ens3_heuristic_tube", "ens3_ensvar_0p5", "ens3_gen_v3_edit"]),
        ("Régime M=5", ["ens5_uniform_baseline", "ens5_heuristic_tube", "ens5_ensvar_0p5", "ens5_gen_v3_edit"]),
    ]
    
    for idx, (title, var_list) in enumerate(comparisons):
        ax = axes[idx]
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Error Quantile", fontsize=12, fontweight="bold")
        ax.set_ylabel("RMSE (Final Round)", fontsize=12, fontweight="bold")
        
        x_positions = np.arange(len(q_keys))
        
        for v in var_list:
            if v not in data:
                continue
            means = []
            stds = []
            for _, qk in q_keys:
                if qk not in data[v] or not data[v][qk]:
                    continue
                max_r = max(data[v][qk].keys())
                vals = data[v][qk][max_r]
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            if len(means) != len(q_keys):
                continue
                
            means = np.array(means)
            stds = np.array(stds)
            color = get_color_for_variant(v)
            label = format_variant_name(v).split(" (")[0]
            ls = "--" if "uniform" in v else ("-." if "heuristic" in v else "-")
            marker = "o" if "uniform" in v else ("^" if "heuristic" in v else "s")
            
            ax.plot(x_positions, means, label=label, color=color, linestyle=ls, marker=marker, linewidth=2.2)
            ax.fill_between(x_positions, np.maximum(0, means - stds), means + stds, color=color, alpha=0.15)
            
        ax.set_xticks(x_positions)
        ax.set_xticklabels(q_labels)
        ax.legend(loc="upper left", framealpha=0.85, fontsize=8.5)
        
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path = out_dir / "fig3_error_quantiles_spectrum.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved {out_path}")

# -----------------------------------------------------------------------------
# Figure 4: Long-Horizon Rollout Robustness (Mean vs Median & IQR)
# -----------------------------------------------------------------------------
def plot_figure_4_rollout(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True, dpi=200)
    fig.suptitle("Figure 4 : Long-Horizon Rollout Robustness (t=100 Steps, Final Round, Log Scale)", fontsize=16, fontweight="bold", y=0.98)
    
    m_configs = [
        (1, ["ens1_uniform_baseline", "ens1_heuristic_tube", "pure_scratch_loss_spectrum", "ens1_loss_edit_smooth", "ens1_loss_jitter_replay", "ens1_loss_sobolev_tv"]),
        (3, ["ens3_uniform_baseline", "ens3_heuristic_tube", "ens3_ensvar_0p5", "ens3_gen_v3_edit"]),
        (5, ["ens5_uniform_baseline", "ens5_heuristic_tube", "ens5_ensvar_0p5", "ens5_gen_v3_edit"]),
    ]
    
    for idx, (m_val, var_list) in enumerate(m_configs):
        ax = axes[idx]
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.set_title(f"Régime M = {m_val}", fontsize=14, fontweight="bold")
        
        x_positions = np.arange(len(var_list))
        labels = []
        
        for i, v in enumerate(var_list):
            labels.append(format_variant_name(v).split(" (")[0])
            if v not in data or "rollout/rmse_final" not in data[v]:
                continue
            max_r = max(data[v]["rollout/rmse_final"].keys())
            f_vals = np.array(data[v]["rollout/rmse_final"][max_r])
            
            med = np.median(f_vals)
            p25, p75 = np.percentile(f_vals, 25), np.percentile(f_vals, 75)
            mean = np.mean(f_vals)
            mn, mx = np.min(f_vals), np.max(f_vals)
            col = get_color_for_variant(v)
            
            # IQR box
            ax.bar(i, p75 - p25, bottom=p25, width=0.5, color=col, alpha=0.35, edgecolor=col, linewidth=2, label="IQR [Q25, Q75]" if i == 0 and idx == 0 else "")
            # Median bar
            ax.plot([i - 0.28, i + 0.28], [med, med], color=col, linewidth=3.5, label="Médiane (p50)" if i == 0 and idx == 0 else "")
            # Mean marker
            ax.scatter(i, mean, color="red", s=55, zorder=5, marker="D", label="Moyenne" if i == 0 and idx == 0 else "")
            # Whiskers
            ax.vlines(i, mn, min(mx, 50.0), color=col, linestyle="--", linewidth=1.5)
            
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9.5)
        if idx == 0:
            ax.set_ylabel("Final Rollout RMSE (t=100)", fontsize=12, fontweight="bold")
            ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 65.0)
        
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path = out_dir / "fig4_rollout_stability.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved {out_path}")

# -----------------------------------------------------------------------------
# Figure 5: Relative Error Reduction vs Control Baseline (% Gain)
# -----------------------------------------------------------------------------
def plot_figure_5_relative_gains(data, out_dir):
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=200)
    ax.set_title("Figure 5 : Relative Error Reduction vs Uniform Control at Equal M (Tube Mid-k ρ=0.5)", fontsize=15, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    comparisons = [
        ("ens1_heuristic_tube", "ens1_uniform_baseline", "Heuristic Random (M=1)", COLORS["random_tube"]),
        ("pure_scratch_loss_spectrum", "ens1_uniform_baseline", "Loss-Spectrum (M=1)", COLORS["spectrum"]),
        ("ens1_loss_edit_smooth", "ens1_uniform_baseline", "Edit Smooth (M=1)", COLORS["gen_edit"]),
        ("ens1_loss_jitter_replay", "ens1_uniform_baseline", "Jitter Replay (M=1)", COLORS["jitter"]),
        ("ens1_loss_sobolev_tv", "ens1_uniform_baseline", "Sobolev H¹ (M=1)", COLORS["sobolev"]),
        ("ens3_heuristic_tube", "ens3_uniform_baseline", "Heuristic Random (M=3)", COLORS["random_tube"]),
        ("ens3_ensvar_0p5", "ens3_uniform_baseline", "EnsVar 0.5 (M=3)", COLORS["ensvar"]),
        ("ens3_gen_v3_edit", "ens3_uniform_baseline", "Gen-V3 (M=3)", COLORS["gen_edit"]),
        ("ens5_heuristic_tube", "ens5_uniform_baseline", "Heuristic Random (M=5)", COLORS["random_tube"]),
        ("ens5_ensvar_0p5", "ens5_uniform_baseline", "EnsVar 0.5 (M=5)", COLORS["ensvar"]),
        ("ens5_gen_v3_edit", "ens5_uniform_baseline", "Gen-V3 (M=5)", COLORS["gen_edit"]),
    ]
    
    mkey = "hard_val/val_tube_midk_rho0p5/rmse_mean"
    labels = []
    gains = []
    gain_stds = []
    bar_colors = []
    
    for var, base, label, col in comparisons:
        if var not in data or base not in data or mkey not in data[var] or mkey not in data[base]:
            continue
        max_r_var = max(data[var][mkey].keys())
        max_r_base = max(data[base][mkey].keys())
        
        v_vals = np.array(data[var][mkey][max_r_var])
        b_vals = np.array(data[base][mkey][max_r_base])
        
        min_len = min(len(v_vals), len(b_vals))
        paired_gains = 100.0 * (b_vals[:min_len] - v_vals[:min_len]) / b_vals[:min_len]
        
        labels.append(label)
        gains.append(np.mean(paired_gains))
        gain_stds.append(np.std(paired_gains) if len(paired_gains) > 1 else 0.0)
        bar_colors.append(col)
        
    x = np.arange(len(labels))
    bars = ax.bar(x, gains, yerr=gain_stds, color=bar_colors, alpha=0.88, edgecolor="black", linewidth=1.1, capsize=4)
    
    for bar, gain in zip(bars, gains):
        yval = bar.get_height()
        sign = "+" if gain >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (1.5 if gain >= 0 else -4.0), f"{sign}{gain:.1f}%", ha="center", va="bottom" if gain >= 0 else "top", fontsize=10, fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("Réduction d'Erreur vs Contrôle (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(-10, 105)
    ax.axhline(0, color="black", linewidth=1)
    
    plt.tight_layout()
    out_path = out_dir / "fig5_relative_gains.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved {out_path}")

def main():
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data, variant_seeds = load_all_campaign_data()
    print(f"Loaded data for {len(data)} variants across {sum(len(s) for s in variant_seeds.values())} seed runs.")
    
    plot_figure_1_convergence(data, out_dir)
    plot_figure_2_rho_sensitivity(data, out_dir)
    plot_figure_3_quantiles(data, out_dir)
    plot_figure_4_rollout(data, out_dir)
    plot_figure_5_relative_gains(data, out_dir)
    print("\n[✓] All publication figures 1 to 5 regenerated successfully in docs/figures/")

if __name__ == "__main__":
    main()
