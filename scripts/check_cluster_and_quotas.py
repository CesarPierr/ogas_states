#!/usr/bin/env python3
import subprocess
import shutil

print("=== 1. SLURM QUEUE REASONS ===")
try:
    res = subprocess.run(["squeue", "-u", "pcesar00", "-o", "%.10i %.20j %.8T %.10M %.25r"], capture_output=True, text=True, timeout=10)
    lines = res.stdout.strip().split("\n")
    print(f"Total entries in queue: {len(lines)-1 if len(lines)>1 else 0}")
    for l in lines[:15]:
        print(" ", l)
except Exception as e:
    print("Error querying squeue:", e)

print("\n=== 2. DETAILED JOB REASON (Sample Job) ===")
try:
    res = subprocess.run(["squeue", "-u", "pcesar00", "-h", "-o", "%i"], capture_output=True, text=True, timeout=10)
    job_ids = res.stdout.strip().split()
    if job_ids:
        jid = job_ids[0]
        res_ctrl = subprocess.run(["scontrol", "show", "job", jid], capture_output=True, text=True, timeout=10)
        for l in res_ctrl.stdout.split("\n"):
            if any(k in l for k in ["JobState", "Reason", "QOS", "Account", "NumNodes", "ReqTRES", "SubmitTime", "StartTime", "Priority"]):
                print(" ", l.strip())
except Exception as e:
    print("Error querying scontrol:", e)

print("\n=== 3. ACCOUNT BUDGET & QUOTA (saldo / cindata) ===")
if shutil.which("saldo"):
    try:
        r_saldo = subprocess.run(["saldo", "-b"], capture_output=True, text=True, timeout=15)
        print("--- saldo -b ---")
        print(r_saldo.stdout if r_saldo.stdout else r_saldo.stderr)
    except Exception as e:
        print("saldo error:", e)

if shutil.which("cindata"):
    try:
        r_cin = subprocess.run(["cindata"], capture_output=True, text=True, timeout=15)
        print("--- cindata ---")
        print(r_cin.stdout if r_cin.stdout else r_cin.stderr)
    except Exception as e:
        print("cindata error:", e)

print("\n=== 4. SCRATCH DISK SPACE ===")
try:
    r_df = subprocess.run(["df", "-h", "/leonardo_scratch/fast/EUHPC_D36_033"], capture_output=True, text=True, timeout=5)
    print(r_df.stdout)
except Exception as e:
    print("df error:", e)
