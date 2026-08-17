import json
import numpy as np

with open("/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_v3_pilot/campaign_analysis_rmse_mean.json") as f:
    d_rmse = json.load(f)
with open("/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_v3_pilot/campaign_analysis_rmse_p99.json") as f:
    d_p99 = json.load(f)
with open("/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_v3_pilot/campaign_analysis_nrmse_mean.json") as f:
    d_nrmse = json.load(f)

arms = ["uniform_baseline", "gen_pure_edit", "random_tube", "tube_select", "mined_ic", "gen_v3", "gen_v3_edit", "gen_v3_guard"]
metrics = {
    "Bulk (RMSE)": ("val/rmse_mean", d_rmse),
    "Bulk (NRMSE)": ("val/nrmse_mean", d_nrmse),
    "TV-Hard (p99 RMSE)": ("hard_val/val_hard_tv_div/rmse_p99", d_p99),
    "Tube low-k ρ=0.1": ("hard_val/val_tube_lowk_rho0p1/nrmse_mean", d_nrmse),
    "Tube low-k ρ=0.5": ("hard_val/val_tube_lowk_rho0p5/nrmse_mean", d_nrmse),
    "Tube mid-k ρ=0.1": ("hard_val/val_tube_midk_rho0p1/nrmse_mean", d_nrmse),
    "Tube mid-k ρ=0.5": ("hard_val/val_tube_midk_rho0p5/nrmse_mean", d_nrmse)
}

print(f"| Variante d'Échantillonnage (Arm) | " + " | ".join(metrics.keys()) + " |")
print("| :--- | " + " | ".join([":---:"] * len(metrics)) + " |")

# Extract uniform_baseline means
ref_means = {}
for m_name, (m_key, d) in metrics.items():
    # To get uniform_baseline mean, find any row where vs == 'uniform_baseline'
    row = next((r for r in d["rows"] if r["vs"] == "uniform_baseline" and r["metric"] == m_key), None)
    if row:
        ref_means[m_name] = row["ref_mean"]
    else:
        # Fallback if vs == 'uniform_baseline' isn't available
        row = next((r for r in d["rows"] if r["arm"] == "uniform_baseline" and r["metric"] == m_key), None)
        if row:
            ref_means[m_name] = row["mean"]

for arm in arms:
    row_str = f"| **{arm}** |"
    for m_name, (m_key, d) in metrics.items():
        # Just grab the row for `arm` to get its absolute mean
        row = next((r for r in d["rows"] if r["arm"] == arm and r["metric"] == m_key), None)
        if row:
            mean = row["mean"]
            std = row["std"]
            ref_mean = ref_means.get(m_name, mean)
            if arm == "uniform_baseline":
                pct_str = ""
            else:
                pct = 100 * (mean - ref_mean) / (abs(ref_mean) + 1e-12)
                pct_str = f" ({pct:+.1f}%)"
            val_str = f"{mean:.4f} ± {std:.4f}"
            row_str += f" {val_str}{pct_str} |"
        else:
            row_str += f" MISSING |"
    print(row_str)
