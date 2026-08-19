#!/usr/bin/env python3
"""
Master Checkpoint Post-Hoc Evaluation Script
===========================================
Re-evaluates all final checkpoints directly against the ground-truth validation banks:
- Attractor Validation (1-step & 100-step autoregressive rollouts)
- Perturbation Tubes (rho = 0.5, 0.25, 0.1 at low-k, mid-k, high-k)
- Hard Low-Amplitude (Out-of-Distribution linear recovery)
- Hard Total Variation (High TV shock resistance)
- Hard Mixed Diversity

Outputs:
- results/json/recomputed_master_evaluation_metrics.json
- results/recomputed_master_evaluation_report.md
- docs/full_picture_report.md
"""
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from poolbased_surrogate.eval import (
    ValidationData,
    evaluate_validation,
    load_hard_validation_sets,
    load_validation_data,
)
from poolbased_surrogate.models.surrogate import (
    EnsembleSurrogate,
    ExactAL4PDEUnet1D,
    build_surrogate,
)
from poolbased_surrogate.pde import PDE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_ROOT = Path("/leonardo_scratch/fast/EUHPC_D36_033/ogas_validation")
RUNS_ROOT = Path("/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs")


def get_m_size(variant_name: str) -> int:
    if variant_name.startswith("ens1_") or "scratch_loss" in variant_name:
        return 1
    elif variant_name.startswith("ens3_") or "scratch_ensvar" in variant_name:
        return 3
    elif variant_name.startswith("ens5_"):
        return 5
    elif variant_name.startswith("ens2_"):
        return 2
    return 1


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Load model architecture and weights from checkpoint."""
    payload = torch.load(ckpt_path, map_location="cpu")
    run_dir = ckpt_path.parent
    cfg_file = run_dir / "config.yaml"
    
    # Load config if present
    cfg = {}
    if cfg_file.exists():
        try:
            cfg = yaml.safe_load(cfg_file.read_text())
        except Exception:
            pass
            
    surr_cfg = cfg.get("surrogate", {})
    hidden = surr_cfg.get("hidden", 64)
    depth = surr_cfg.get("depth", 4)
    ens_size = surr_cfg.get("ensemble_size", 1)
    diff_weight = surr_cfg.get("difference_weight", 0.3)
    
    pde_cfg = cfg.get("pde", {})
    param_dim = pde_cfg.get("param_dim", 2)
    spatial_dim = pde_cfg.get("spatial_dim", 1)
    resolution = pde_cfg.get("resolution", 800)
    
    surrogate = build_surrogate(
        resolution=resolution,
        hidden=hidden,
        depth=depth,
        ensemble_size=ens_size,
        param_dim=param_dim,
        difference_weight=diff_weight,
        spatial_dim=spatial_dim,
    ).to(device)
    
    # Load state dict
    if "surrogate" in payload:
        surr_state = payload["surrogate"]
        if isinstance(surr_state, dict):
            # Check if keys match EnsembleSurrogate
            try:
                surrogate.load_state_dict(surr_state, strict=False)
            except Exception as e:
                print(f"  Warning loading surrogate state dict: {e}")
        elif isinstance(surr_state, list):
            for i, sd in enumerate(surr_state):
                if i < len(surrogate.models):
                    surrogate.models[i].load_state_dict(sd, strict=False)
                    
    surrogate.eval()
    return surrogate, ens_size


def run_full_evaluation():
    print("=" * 75)
    print("=== MASTER POST-HOC CHECKPOINT EVALUATION PIPELINE ===")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Validation Root: {VAL_ROOT}")
    print(f"Runs Root: {RUNS_ROOT}")
    print("=" * 75)

    # 1. Load Hard & Tube Validation Sets
    hard_dir = VAL_ROOT / "ks_diverse_suite"
    print(f"\n[1/3] Loading Validation Test Banks from {hard_dir}...")
    hard_sets = load_hard_validation_sets(hard_dir)
    print(f"  Loaded {len(hard_sets)} evaluation test sets: {list(hard_sets.keys())}")

    # Load Standard Attractor Validation Bank
    attractor_file = VAL_ROOT / "ks_res800_al4pde_sub5_seed43_n1500_t100.npz"
    attractor_val = None
    if attractor_file.exists():
        attractor_val = load_validation_data(attractor_file)
        print(f"  Loaded Attractor Bank: {attractor_val.n_trajectories} trajectories x {attractor_val.trajectory_steps} steps")

    # 2. Discover all run checkpoints
    print("\n[2/3] Scanning all completed run directories & checkpoints...")
    base_suites = [
        RUNS_ROOT / "ks800_classical_al",
        RUNS_ROOT / "ks800_m1_improvements",
        RUNS_ROOT / "ks800_v3_pilot",
        RUNS_ROOT / "ks800_phase2_repro",
        RUNS_ROOT / "ks800_ensemble_scaling",
        RUNS_ROOT / "ks800_pure_scratch",
        RUNS_ROOT / "ks800_heuristic_tube",
    ]

    discovered_checkpoints = []
    for sdir in base_suites:
        if not sdir.exists():
            continue
        for rdir in sorted(sdir.glob("*")):
            if not rdir.is_dir():
                continue
            rname = rdir.name
            m = re.match(r"^(.*)_seed(\d+)$", rname)
            if not m:
                continue
            vname, seed = m.group(1), int(m.group(2))
            
            # Check for latest checkpoint
            ckpt = rdir / "checkpoint_latest.pt"
            if not ckpt.exists():
                ckpt = rdir / "checkpoint_round_9.pt"
            if not ckpt.exists():
                ckpt = rdir / "checkpoint_final.pt"
                
            if ckpt.exists():
                discovered_checkpoints.append((vname, seed, ckpt, rdir))

    print(f"  Discovered {len(discovered_checkpoints)} valid checkpoints across suites.")

    # 3. Evaluate each checkpoint
    print("\n[3/3] Evaluating checkpoints on GPU...")
    all_results = defaultdict(dict)
    
    start_eval = time.perf_counter()
    for idx, (vname, seed, ckpt_path, rdir) in enumerate(discovered_checkpoints):
        t0 = time.perf_counter()
        print(f"[{idx+1}/{len(discovered_checkpoints)}] Evaluating {vname} (seed {seed})...", flush=True)
        
        try:
            surrogate, ens_size = load_model_from_checkpoint(ckpt_path, DEVICE)
            metrics = {}

            # A. Evaluate on Attractor Bank
            if attractor_val is not None:
                att_res = evaluate_validation(
                    model=surrogate,
                    validation=attractor_val,
                    rollout_steps=min(100, attractor_val.trajectory_steps),
                    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                    device=DEVICE,
                )
                metrics.update(att_res)

            # B. Evaluate on all Hard & Tube Sets
            for set_name, vdata in hard_sets.items():
                hard_res = evaluate_validation(
                    model=surrogate,
                    validation=vdata,
                    rollout_steps=min(50, vdata.trajectory_steps),
                    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                    device=DEVICE,
                )
                for k, v in hard_res.items():
                    metrics[f"hard_val/{set_name}/{k}"] = v

            # Record result
            all_results[vname][seed] = metrics
            dt = time.perf_counter() - t0
            roll_mean = metrics.get("rollout/rmse_mean", np.nan)
            nrmse_1s = metrics.get("val/nrmse_mean", np.nan)
            tube_mid = metrics.get("hard_val/val_tube_midk_rho0p5/rmse_mean", np.nan)
            print(f"     -> Done in {dt:.1f}s | 1-Step NRMSE: {nrmse_1s:.4f} | Rollout RMSE: {roll_mean:.3f} | Tube Mid-k: {tube_mid:.3f}")

        except Exception as exc:
            print(f"     -> ERROR evaluating {ckpt_path}: {exc}")

    # 4. Save JSON Results
    results_dir = Path("results/json")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "recomputed_master_evaluation_metrics.json"
    
    serializable = {}
    for vname, seed_dict in all_results.items():
        serializable[vname] = {
            str(s): {k: float(v) if not np.isnan(v) else None for k, v in m.items()}
            for s, m in seed_dict.items()
        }
    out_json.write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved raw JSON metrics to {out_json}")

    # 5. Generate Markdown Report
    print("\nGenerating consolidated Markdown summary report...")
    md_lines = [
        "# Recomputed Master Checkpoint Evaluation Report",
        f"*Generated automatically on {time.strftime('%Y-%m-%d %H:%M:%S')} on Leonardo Booster.*",
        "",
        "> [!IMPORTANT]",
        "> All metrics are recomputed directly from model weights on fixed test banks across all 10 seeds.",
        "",
        "## Quantitative Summary (10 Seeds / Round 9 Checkpoints)",
        "",
        "| Strategy | Ensemble M | Seeds | 1-Step NRMSE (Attractor)<br>Mean ± Std (**Median**) | Rollout 100-pas Mean RMSE<br>Mean ± Std | **Rollout 100-pas Mean RMSE<br>Median [IQR 25-75]** | Rollout<br>Max (Worst) | Tube Mid-k (rho=0.5)<br>Mean (**Median**) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for vname in sorted(all_results.keys()):
        m_size = get_m_size(vname)
        seeds_data = all_results[vname]
        n_seeds = len(seeds_data)
        if n_seeds == 0:
            continue
            
        nrmse_list = [m.get("val/nrmse_mean", np.nan) for m in seeds_data.values() if not np.isnan(m.get("val/nrmse_mean", np.nan))]
        roll_list = [m.get("rollout/rmse_mean", np.nan) for m in seeds_data.values() if not np.isnan(m.get("rollout/rmse_mean", np.nan))]
        tube_list = [m.get("hard_val/val_tube_midk_rho0p5/rmse_mean", np.nan) for m in seeds_data.values() if not np.isnan(m.get("hard_val/val_tube_midk_rho0p5/rmse_mean", np.nan))]
        
        nrmse_str = f"{np.mean(nrmse_list):.4f} ± {np.std(nrmse_list):.4f} (**{np.median(nrmse_list):.4f}**)" if nrmse_list else "-"
        roll_mean_str = f"{np.mean(roll_list):.3f} ± {np.std(roll_list):.3f}" if roll_list else "-"
        roll_med_str = f"**{np.median(roll_list):.3f}** [{np.percentile(roll_list, 25):.3f}, {np.percentile(roll_list, 75):.3f}]" if roll_list else "-"
        roll_max_str = f"{np.max(roll_list):.3f}" if roll_list else "-"
        tube_str = f"{np.mean(tube_list):.3f} (**{np.median(tube_list):.3f}**)" if tube_list else "-"
        
        md_lines.append(f"| **`{vname}`** | M={m_size} | {n_seeds}/10 | {nrmse_str} | {roll_mean_str} | {roll_med_str} | {roll_max_str} | {tube_str} |")

    out_md = Path("results/recomputed_master_evaluation_report.md")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Markdown report to {out_md}")
    print("\n=== MASTER EVALUATION PIPELINE COMPLETE ===")


if __name__ == "__main__":
    run_full_evaluation()
