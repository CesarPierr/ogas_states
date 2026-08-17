import json
from pathlib import Path

# Mapping to make names more explicit
NAMES_MAP = {
    "uniform_baseline": "Uniform Baseline (contrôle, α=1)",
    "tophalf_0p0": "α=0.0 (tophalf)",
    "tophalf_0p25": "α=0.25 (tophalf)",
    "tophalf_0p5": "α=0.5 (tophalf)",
    "tophalf_0p75": "α=0.75 (tophalf)",
    "ensvar_0p5": "α=0.5 (ensvar)",
}

def get_name(raw_name):
    return NAMES_MAP.get(raw_name, raw_name)

def generate_table_v2(json_1step: Path, json_rollout: Path):
    with open(json_1step) as f:
        data_1step = json.load(f)
    
    with open(json_rollout) as f:
        data_rollout = json.load(f)
    
    # Specific columns
    columns_1step = [
        ("Uniform (mean RMSE)", "rmse_mean", "val_unif"),
        ("HARD amplitudes", "rmse_mean", "val_hard_tv_div"),
        ("Tube (ρ=0.1)", "rmse_mean", "val_tube_midk_rho0p1"),
        ("Tube (ρ=0.5)", "rmse_mean", "val_tube_midk_rho0p5"),
    ]
    
    columns_rollout = [
        ("Rollout @10 steps (NRMSE)", "nrmse@10"),
        ("Rollout @50 steps (NRMSE)", "nrmse@50"),
        ("Rollout @100 steps (NRMSE)", "nrmse@final"),
    ]
    
    parsed = {}
    
    # 1. Parse 1-step metrics
    for comp in data_1step["comparisons"]:
        if comp["baseline"] != "uniform_baseline":
            continue
        v = comp["variant"]
        m = comp["metric"]
        s = comp["set"]
        
        if v not in parsed:
            parsed[v] = {}
        parsed[v][(m, s)] = comp["variant_mean"]
        
        bl = comp["baseline"]
        if bl not in parsed:
            parsed[bl] = {}
        parsed[bl][(m, s)] = comp["ref_mean"]

    # 2. Parse rollout metrics (we average over seeds inside the script if needed, or maybe it's already aggregated? No, it's a list of dicts per variant)
    for variant, metrics_list in data_rollout.items():
        if variant not in parsed:
            parsed[variant] = {}
        # compute mean across seeds
        for m_name in [c[1] for c in columns_rollout]:
            vals = [m[m_name] for m in metrics_list if m_name in m]
            if vals:
                parsed[variant][m_name] = sum(vals) / len(vals)

    md = ["## Vague 2 (Re-évaluation)"]
    
    header_cols = [c[0] for c in columns_1step] + [c[0] for c in columns_rollout]
    header = "| Sampling Strategy | " + " | ".join(header_cols) + " |"
    separator = "|---|---|" + "|".join(["---" for _ in header_cols[1:]]) + "|"
    md.append(header)
    md.append(separator)
    
    order = ["uniform_baseline", "tophalf_0p0", "tophalf_0p25", "tophalf_0p5", "tophalf_0p75", "ensvar_0p5"]
    
    for v in order:
        if v not in parsed:
            continue
        name = get_name(v)
        row = f"| {name} |"
        
        # 1-step columns
        for col_name, m, s in columns_1step:
            if (m, s) in parsed.get(v, {}) and "uniform_baseline" in parsed and (m, s) in parsed["uniform_baseline"]:
                v_mean = parsed[v][(m, s)]
                ref_mean = parsed["uniform_baseline"][(m, s)]
                
                if v == "uniform_baseline":
                    row += f" {v_mean:.4f} |"
                else:
                    pct_change = 100.0 * (v_mean - ref_mean) / ref_mean
                    sign = "+" if pct_change > 0 else ""
                    row += f" {v_mean:.4f} ({sign}{pct_change:.1f}%) |"
            else:
                row += " N/A |"
                
        # Rollout columns
        for col_name, m in columns_rollout:
            if m in parsed.get(v, {}) and "uniform_baseline" in parsed and m in parsed["uniform_baseline"]:
                v_mean = parsed[v][m]
                ref_mean = parsed["uniform_baseline"][m]
                
                if v == "uniform_baseline":
                    row += f" {v_mean:.3f} |"
                else:
                    pct_change = 100.0 * (v_mean - ref_mean) / ref_mean
                    sign = "+" if pct_change > 0 else ""
                    row += f" {v_mean:.3f} ({sign}{pct_change:.1f}%) |"
            else:
                row += " N/A |"
                
        md.append(row)
    md.append("")
    
    return "\n".join(md)

def main():
    v2_1step = Path("campaign_detailed_posthoc_analysis_v2.json")
    v2_rollout = Path("rollout_analysis_v2.json")
    v2_ens3_1step = Path("campaign_detailed_posthoc_analysis_v2_ens3.json")
    v2_ens3_rollout = Path("rollout_analysis_v2_ens3.json")
    
    md_content = "# Résultats de la Vague 2 (Phase 2)\n\n"
    
    if v2_1step.exists() and v2_rollout.exists():
        md_content += generate_table_v2(v2_1step, v2_rollout)
    else:
        md_content += "Les fichiers JSON d'analyse n'ont pas encore été générés."

    md_content += "\n\n# Résultats de la Vague 2 avec ensemble_size=3\n\n"
    
    if v2_ens3_1step.exists() and v2_ens3_rollout.exists():
        md_content += generate_table_v2(v2_ens3_1step, v2_ens3_rollout)
    else:
        md_content += "Les fichiers JSON d'analyse ens3 n'ont pas encore été générés."
        
    out_path = Path("docs/res.md")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(md_content)
    print(f"Written tables to {out_path}")

if __name__ == "__main__":
    main()
