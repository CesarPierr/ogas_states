#!/usr/bin/env python3
"""
Tube Perturbations Spectrum Evaluator across rho in {0.1, 0.25, 0.5} and k in {low, mid, high}
=============================================================================================
Evaluates all 50 Burgers Profile B checkpoints on:
- Tube rho=0.10 (low-k, mid-k, high-k) -> Near-attractor exploration
- Tube rho=0.25 (low-k, mid-k, high-k) -> Moderate out-of-distribution
- Tube rho=0.50 (low-k, mid-k, high-k) -> Extreme out-of-distribution tube boundary

Generates detailed breakdown table across rho and k-modes.
"""
import glob
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from poolbased_surrogate.eval import ValidationData, evaluate_validation, load_validation_data
from poolbased_surrogate.models.surrogate import build_surrogate
from poolbased_surrogate.pde import ensure_al4pde_paths

ensure_al4pde_paths()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_ROOT = Path("/leonardo_scratch/fast/EUHPC_D36_033/ogas_validation/burgers_hard2_suite")
RUNS_ROOT = Path("/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/burgers_profile_b_sweep")

TUBE_SETS = [
    ("val_tube_lowk_rho0p1", "rho=0.10 (low-k)"),
    ("val_tube_midk_rho0p1", "rho=0.10 (mid-k)"),
    ("val_tube_highk_rho0p1", "rho=0.10 (high-k)"),
    ("val_tube_lowk_rho0p25", "rho=0.25 (low-k)"),
    ("val_tube_midk_rho0p25", "rho=0.25 (mid-k)"),
    ("val_tube_highk_rho0p25", "rho=0.25 (high-k)"),
    ("val_tube_lowk_rho0p5", "rho=0.50 (low-k)"),
    ("val_tube_midk_rho0p5", "rho=0.50 (mid-k)"),
    ("val_tube_highk_rho0p5", "rho=0.50 (high-k)"),
]

STRATEGIES = [
    "uniform_baseline",
    "heuristic_tube",
    "classic_al_topk",
    "classic_al_sbal",
    "ogas_generative",
]


def load_model(ckpt_path: Path):
    payload = torch.load(ckpt_path, map_location="cpu")
    run_dir = ckpt_path.parent
    cfg_file = run_dir / "config.resolved.json"
    
    cfg = {}
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text())
        except Exception:
            pass
            
    surr_cfg = cfg.get("surrogate", {})
    hidden = surr_cfg.get("hidden", 64)
    depth = surr_cfg.get("depth", 4)
    ens_size = surr_cfg.get("ensemble_size", 2)
    diff_weight = surr_cfg.get("difference_weight", 0.3)
    
    surrogate = build_surrogate(
        resolution=256,
        hidden=hidden,
        depth=depth,
        ensemble_size=ens_size,
        param_dim=1,
        difference_weight=diff_weight,
        spatial_dim=1,
    ).to(DEVICE)
    
    if "surrogate" in payload:
        surr_state = payload["surrogate"]
        if isinstance(surr_state, dict):
            surrogate.load_state_dict(surr_state)
        elif isinstance(surr_state, list):
            for m, s in zip(surrogate.models, surr_state):
                m.load_state_dict(s)
    surrogate.eval()
    return surrogate


def main():
    print("=" * 110)
    print("=== EVALUATION OF TUBE PERTURBATIONS: RHO in {0.10, 0.25, 0.50} & K in {LOW, MID, HIGH} ===")
    print(f"Device: {DEVICE}")
    print("=" * 110)

    # 1. Load Tube Validation Sets
    loaded_tubes = {}
    for set_key, label in TUBE_SETS:
        p = VAL_ROOT / f"{set_key}.npz"
        if p.exists():
            vdata = load_validation_data(p)
            if vdata.states0.shape[-1] == 512:
                vdata = ValidationData(
                    states0=vdata.states0[..., ::2],
                    params=vdata.params,
                    trajectories=vdata.trajectories[..., ::2],
                )
            loaded_tubes[set_key] = (label, vdata)
            print(f"Loaded {set_key} ({label}): shape={vdata.trajectories.shape}")

    # 2. Discover Checkpoints
    discovered = defaultdict(dict)
    for strat in STRATEGIES:
        for rdir in sorted(RUNS_ROOT.glob(f"{strat}_seed*")):
            m = re.match(r"^.*_seed(\d+)$", rdir.name)
            if not m:
                continue
            seed = int(m.group(1))
            ckpt = rdir / "checkpoint_latest.pt"
            if not ckpt.exists():
                ckpt = rdir / "surrogate.pt"
            if ckpt.exists():
                discovered[strat][seed] = ckpt

    # 3. Evaluate each strategy on each tube set
    results_by_rho = defaultdict(lambda: defaultdict(list))
    
    print("\nEvaluating all checkpoints across tube sets...")
    for strat, seed_dict in discovered.items():
        print(f"\n--- Strategy: {strat} ({len(seed_dict)} seeds) ---")
        for seed, ckpt in sorted(seed_dict.items()):
            try:
                model = load_model(ckpt)
                for set_key, (label, vdata) in loaded_tubes.items():
                    res = evaluate_validation(
                        model=model,
                        validation=vdata,
                        rollout_steps=min(40, vdata.trajectory_steps),
                        quantiles=[0.1, 0.5, 0.9, 0.95],
                        device=DEVICE,
                    )
                    nrmse_1s = res.get("val/nrmse_mean", np.nan)
                    nrmse_roll = res.get("rollout/nrmse_mean", np.nan)
                    results_by_rho[strat][set_key].append({
                        "1s": nrmse_1s,
                        "roll": nrmse_roll,
                    })
            except Exception as e:
                print(f"Error evaluating {strat} seed {seed}: {e}")

    # 4. Generate Aggregated Summary Table
    print("\n" + "=" * 120)
    print("SUMMARY OF TUBE PERTURBATION PERFORMANCE (NRMSE 1-STEP & ROLLOUT T=40)")
    print("=" * 120)
    
    md_lines = [
        "# Tube Perturbation Robustness: Analysis Across $\\rho \\in \\{0.10, 0.25, 0.50\\}$ & Frequencies",
        "",
        "| Strategy | $\\rho=0.10$ (Low-k) | $\\rho=0.10$ (Mid-k) | $\\rho=0.10$ (High-k) | $\\rho=0.25$ (Mid-k) | $\\rho=0.50$ (Low-k) | $\\rho=0.50$ (Mid-k) | $\\rho=0.50$ (High-k) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    
    for strat in STRATEGIES:
        strat_data = results_by_rho[strat]
        cells = []
        for set_key in ["val_tube_lowk_rho0p1", "val_tube_midk_rho0p1", "val_tube_highk_rho0p1",
                        "val_tube_midk_rho0p25", "val_tube_lowk_rho0p5", "val_tube_midk_rho0p5", "val_tube_highk_rho0p5"]:
            records = strat_data.get(set_key, [])
            if records:
                rolls = [r["roll"] for r in records if not np.isnan(r["roll"])]
                if rolls:
                    cells.append(f"{np.mean(rolls):.4f} ± {np.std(rolls):.4f}")
                else:
                    cells.append("N/A")
            else:
                cells.append("N/A")
        md_lines.append(f"| **{strat}** | " + " | ".join(cells) + " |")

    out_md = Path("docs/burgers_tube_rho_analysis.md")
    out_md.write_text("\n".join(md_lines))
    print("\n".join(md_lines))
    print(f"\n[REPORT SAVED] Saved tube robustness breakdown to {out_md}")


if __name__ == "__main__":
    main()
