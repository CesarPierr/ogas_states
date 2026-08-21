#!/usr/bin/env bash
# Auto-launcher: monitors Leonardo cluster and submits the 50-job Burgers Profile B sweep as soon as compute nodes are online.

echo "=== Starting Leonardo Cluster Monitor for Burgers Profile B Sweep ==="
while true; do
    # Check if boost_usr_prod has nodes available (> 0)
    avail_nodes=$(sinfo -p boost_usr_prod -h -o "%D" 2>/dev/null || echo "0")
    
    if [ "$avail_nodes" -gt 0 ] 2>/dev/null; then
        echo "[$(date)] Compute nodes available on boost_usr_prod ($avail_nodes nodes). Launching sweeps..."
        echo "=== 1. LAUNCHING 1D BURGERS PROFILE B (50 JOBS) ==="
        bash ./scripts/launch_burgers_profile_b_10seeds.sh || true
        echo "=== 2. LAUNCHING 2D SCALED NAVIER-STOKES (25 JOBS) ==="
        bash ./scripts/launch_2d_ns_scaled_sweep.sh || true
        echo "[$(date)] All sweeps successfully submitted!"
        break
    else
        echo "[$(date)] boost_usr_prod currently in maintenance / 0 nodes. Checking again in 60s..."
        sleep 60
    fi
done
