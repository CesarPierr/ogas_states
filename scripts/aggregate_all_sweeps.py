#!/usr/bin/env python3
"""
Comprehensive Benchmark Results Aggregator
==========================================
Aggregates and formats results for:
1. 1D Burgers Profile B (High-Reynolds, 10 seeds x 5 strategies = 50 jobs)
2. 2D Scaled Navier-Stokes (Kolmogorov Flow, 5 seeds x 5 strategies = 25 jobs)
"""
import glob
import json
import re
from pathlib import Path
import numpy as np

def aggregate_1d_burgers():
    print("=" * 90)
    print("1. BURGERS PROFILE B (HIGH-REYNOLDS Re in [2000, 10000], 10 SEEDS) BENCHMARK")
    print("=" * 90)
    print(f"{'Strategy':22s} | {'Completed':10s} | {'Rollout NRMSE':22s} | {'1-Step NRMSE':22s}")
    print("-" * 90)
    
    base_dir = "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/burgers_profile_b_sweep"
    runs = sorted(glob.glob(f"{base_dir}/*"))
    strats = ["uniform_baseline", "heuristic_tube", "classic_al_topk", "classic_al_sbal", "ogas_generative"]
    
    for s in strats:
        s_runs = [r for r in runs if s in r]
        rollouts = []
        step1s = []
        completed = 0
        
        for r in s_runs:
            h_file = Path(r) / "history.json"
            if h_file.exists():
                try:
                    with open(h_file) as f:
                        h = json.load(f)
                        if len(h) >= 10:
                            completed += 1
                            last = h[-1]
                            if "rollout/nrmse_mean" in last:
                                rollouts.append(last["rollout/nrmse_mean"])
                            if "val/nrmse_mean" in last:
                                step1s.append(last["val/nrmse_mean"])
                except Exception:
                    pass
            else:
                # Check .out files for completed runs
                out_files = sorted(glob.glob(f"{r}/*.out"))
                if out_files:
                    try:
                        content = Path(out_files[-1]).read_text()
                        if "round 9:" in content:
                            matches = re.findall(r'"rollout/nrmse_mean":\s*([\d\.e\-]+)', content)
                            if matches:
                                rollouts.append(float(matches[-1]))
                                completed += 1
                            m_step1 = re.findall(r'"val/nrmse_mean":\s*([\d\.e\-]+)', content)
                            if m_step1:
                                step1s.append(float(m_step1[-1]))
                    except Exception:
                        pass
                        
        if completed > 0:
            r_str = f"{np.mean(rollouts):.4f} +/- {np.std(rollouts):.4f}" if rollouts else "N/A"
            s_str = f"{np.mean(step1s):.4f} +/- {np.std(step1s):.4f}" if step1s else "N/A"
            print(f"{s:22s} | {completed:2d}/10      | {r_str:22s} | {s_str:22s}")
        else:
            print(f"{s:22s} |  0/10      | Pending / Running      | Pending / Running")
            
    print("=" * 90)

def aggregate_2d_ns():
    print("\n" + "=" * 90)
    print("2. 2D NAVIER-STOKES SCALED (KOLMOGOROV FLOW Re in [1000, 3000], 5 SEEDS) BENCHMARK")
    print("=" * 90)
    print(f"{'Strategy':22s} | {'Completed':10s} | {'Uniform T25':20s} | {'Hard T25':20s} | {'Tube T25':20s}")
    print("-" * 90)
    
    base_dir = "/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ns2d_scaled_sweep"
    runs = sorted(glob.glob(f"{base_dir}/*"))
    strats = ["uniform_baseline", "heuristic_tube", "classic_al_topk", "classic_al_sbal", "ogas_generative"]
    
    for s in strats:
        s_runs = [r for r in runs if s in r]
        u_t25 = []
        h_t25 = []
        t_t25 = []
        completed = 0
        
        for r in s_runs:
            h_file = Path(r) / "history.json"
            if h_file.exists():
                try:
                    with open(h_file) as f:
                        h = json.load(f)
                        if len(h) >= 5:
                            completed += 1
                            last = h[-1]
                            u_t25.append(last["uniform"]["nrmse_t25"])
                            h_t25.append(last["hard"]["nrmse_t25"])
                            t_t25.append(last["tube"]["nrmse_t25"])
                except Exception:
                    pass
            else:
                out_files = sorted(glob.glob(f"{r}/*.out"))
                if out_files:
                    try:
                        content = Path(out_files[-1]).read_text()
                        if "--- Round 4" in content:
                            u_match = re.findall(r'\[UNIFORM\].*?T25:\s*([\d\.]+)', content)
                            h_match = re.findall(r'\[HARD\].*?T25:\s*([\d\.]+)', content)
                            t_match = re.findall(r'\[TUBE\].*?T25:\s*([\d\.]+)', content)
                            if u_match and h_match and t_match:
                                u_t25.append(float(u_match[-1]))
                                h_t25.append(float(h_match[-1]))
                                t_t25.append(float(t_match[-1]))
                                completed += 1
                    except Exception:
                        pass
                        
        if completed > 0:
            u_str = f"{np.mean(u_t25):.4f} +/- {np.std(u_t25):.4f}" if u_t25 else "N/A"
            h_str = f"{np.mean(h_t25):.4f} +/- {np.std(h_t25):.4f}" if h_t25 else "N/A"
            t_str = f"{np.mean(t_t25):.4f} +/- {np.std(t_t25):.4f}" if t_t25 else "N/A"
            print(f"{s:22s} | {completed:2d}/5       | {u_str:20s} | {h_str:20s} | {t_str:20s}")
        else:
            print(f"{s:22s} |  0/5       | Pending / Running    | Pending / Running    | Pending / Running")
            
    print("=" * 90)

if __name__ == "__main__":
    aggregate_1d_burgers()
    aggregate_2d_ns()
