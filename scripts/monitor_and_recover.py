#!/usr/bin/env python3
"""
Autonomous Benchmark Monitor & Auto-Recovery Engine
===================================================
Monitors:
1. 1D Burgers Profile B (50 jobs: 5 strategies x 10 seeds)
2. 2D Scaled Navier-Stokes (25 jobs: 5 strategies x 5 seeds)

Features:
- Continuously checks SLURM queue and run folder logs.
- Detects crashes / incomplete runs.
- Automatically cleans and relaunches failed runs.
- Prints live aggregated progress table.
"""
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path
import numpy as np


BURGERS_BASE = Path("/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/burgers_profile_b_sweep")
NS2D_BASE = Path("/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ns2d_scaled_sweep")
ROOT = Path("/leonardo/home/userexternal/pcesar00/ogas_states")

STRATS_1D = ["uniform_baseline", "heuristic_tube", "classic_al_topk", "classic_al_sbal", "ogas_generative"]
SEEDS_1D = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

STRATS_2D = ["uniform_baseline", "heuristic_tube", "classic_al_topk", "classic_al_sbal", "ogas_generative"]
SEEDS_2D = [101, 202, 303, 404, 505]


def get_active_slurm_jobs() -> dict[str, str]:
    """Returns mapping of job_name -> status from squeue."""
    try:
        res = subprocess.run(
            ["squeue", "-u", "pcesar00", "-h", "-o", "%i %j %T"],
            capture_output=True,
            text=True,
            check=True,
        )
        jobs = {}
        for line in res.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                job_id, job_name, status = parts[0], parts[1], parts[2]
                jobs[job_name] = status
        return jobs
    except Exception as e:
        print(f"Error querying squeue: {e}")
        return {}


def is_1d_completed(run_dir: Path) -> bool:
    h_file = run_dir / "history.json"
    if h_file.exists():
        try:
            with open(h_file) as f:
                h = json.load(f)
                return len(h) >= 10
        except Exception:
            pass
    return False


def is_2d_completed(run_dir: Path) -> bool:
    h_file = run_dir / "history.json"
    if h_file.exists():
        try:
            with open(h_file) as f:
                h = json.load(f)
                return len(h) >= 5
        except Exception:
            pass
    return False


def relaunch_1d_job(strat: str, seed: int, run_dir: Path):
    print(f"  [AUTO-RECOVER 1D] Relaunching Burgers Profile B: strategy={strat}, seed={seed}...")
    extra_args = []
    if strat == "uniform_baseline":
        extra_args = ["--set", "pool.uniform_fraction=1.0", "--set", "pool.strategy=generator", "--set", "ddpm.enabled=false"]
    elif strat == "heuristic_tube":
        extra_args = ["--set", "pool.uniform_fraction=0.5", "--set", "pool.strategy=random_tube", "--set", "ddpm.enabled=false"]
    elif strat == "classic_al_topk":
        extra_args = ["--set", "pool.uniform_fraction=0.5", "--set", "pool.strategy=classic_al_topk", "--set", "pool.classic_al_oversample=10", "--set", "ddpm.enabled=false"]
    elif strat == "classic_al_sbal":
        extra_args = ["--set", "pool.uniform_fraction=0.5", "--set", "pool.strategy=classic_al_sbal", "--set", "pool.classic_al_oversample=10", "--set", "pool.sbal_alpha=1.0", "--set", "ddpm.enabled=false"]
    elif strat == "ogas_generative":
        extra_args = ["--set", "pool.uniform_fraction=0.5", "--set", "pool.strategy=generator", "--set", "ddpm.enabled=true"]

    env = os.environ.copy()
    env["LEONARDO_RUN_DIR"] = str(run_dir)
    env["LEONARDO_JOB_NAME"] = f"b_b_{strat[:4]}_{seed}"
    env["LEONARDO_WALLTIME"] = "04:00:00"
    
    cmd = [
        str(ROOT / "scripts/submit_leonardo.sh"),
        "configs/1d_burgers/burgers_profile_b.yaml",
        "--fresh",
        "--set", f"seed={seed}",
        "--set", f"output_dir={run_dir}",
        "--set", "wandb.project=burgers-profile-b-sweep",
        "--set", f"wandb.group={strat}",
        *extra_args,
    ]
    subprocess.run(cmd, cwd=str(ROOT), env=env)


def relaunch_2d_job(strat: str, seed: int, run_dir: Path):
    print(f"  [AUTO-RECOVER 2D] Relaunching 2D NS Scaled: strategy={strat}, seed={seed}...")
    job_name = f"ns2d_{strat[:4]}_{seed}"
    sbatch_content = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=euhpc_d36_033
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output={run_dir}/%j.out
#SBATCH --error={run_dir}/%j.err

set -euo pipefail
cd {ROOT}

export OMP_NUM_THREADS=8
export JAX_PLATFORMS=cuda,cpu

module load profile/deeplrn 2>/dev/null || true
module load cuda/12.3 2>/dev/null || module load cuda/12.1 2>/dev/null || true

export PATH="/leonardo/prod/opt/compilers/cuda/12.3/none/bin:$PATH"
export LD_LIBRARY_PATH="/leonardo/prod/opt/compilers/cuda/12.3/none/lib64:${{LD_LIBRARY_PATH:-}}"

source .venv/bin/activate

python -u scripts/run_2d_navier_stokes_scaled.py \\
    --strategy "{strat}" \\
    --seed {seed} \\
    --rounds 5 \\
    --epochs 10 \\
    --n_traj 32 \\
    --steps 20 \\
    --output_dir "{run_dir}"
"""
    subprocess.run(["sbatch"], input=sbatch_content, text=True, cwd=str(ROOT))


def check_cluster_available() -> bool:
    try:
        res = subprocess.run(
            ["sinfo", "-p", "boost_usr_prod", "-h", "-o", "%D"],
            capture_output=True,
            text=True,
            check=True,
        )
        avail = int(res.stdout.strip()) if res.stdout.strip().isdigit() else 0
        return avail > 0
    except Exception:
        return False


def check_and_recover_all():
    if not check_cluster_available():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] boost_usr_prod currently in maintenance / 0 nodes available. Waiting for next cycle...")
        return

    active_jobs = get_active_slurm_jobs()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Active SLURM jobs count: {len(active_jobs)}")
    
    # 1. Check 1D Burgers
    completed_1d = 0
    running_1d = 0
    recovering_1d = 0
    
    for strat in STRATS_1D:
        for seed in SEEDS_1D:
            run_dir = BURGERS_BASE / f"{strat}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            job_prefix = f"b_b_{strat[:4]}_{seed}"
            is_active = any(job_prefix in j for j in active_jobs)
            
            if is_1d_completed(run_dir):
                completed_1d += 1
            elif is_active:
                running_1d += 1
            else:
                # Crashed or not running
                recovering_1d += 1
                relaunch_1d_job(strat, seed, run_dir)
                time.sleep(0.2)
                
    # 2. Check 2D Navier-Stokes
    completed_2d = 0
    running_2d = 0
    recovering_2d = 0
    
    for strat in STRATS_2D:
        for seed in SEEDS_2D:
            run_dir = NS2D_BASE / f"{strat}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            job_prefix = f"ns2d_{strat[:4]}_{seed}"
            is_active = any(job_prefix in j or f"ns2d_sc_{strat}_{seed}" in j for j in active_jobs)
            
            if is_2d_completed(run_dir):
                completed_2d += 1
            elif is_active:
                running_2d += 1
            else:
                recovering_2d += 1
                relaunch_2d_job(strat, seed, run_dir)
                time.sleep(0.2)
                
    print(f"Status Summary:")
    print(f"  1D Burgers:        Completed={completed_1d:2d}/50 | Running/Queued={running_1d:2d} | Recovered/Relaunched={recovering_1d:2d}")
    print(f"  2D Navier-Stokes:  Completed={completed_2d:2d}/25 | Running/Queued={running_2d:2d} | Recovered/Relaunched={recovering_2d:2d}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=120, help="Interval in seconds between checks")
    args = parser.parse_args()

    print("=== Starting Autonomous Benchmark Monitor & Recovery Daemon ===")
    if args.daemon:
        while True:
            try:
                check_and_recover_all()
            except Exception as e:
                print(f"Error during check cycle: {e}")
            time.sleep(args.interval)
    else:
        check_and_recover_all()


if __name__ == "__main__":
    main()
