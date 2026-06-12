#!/usr/bin/env bash
# Matched-round-10 post-hoc analysis of the Phase 2 KS campaign.
#
# Compares the uniform_fraction sweep {0,.25,.5,.75} (top_half) + uf50_ensvar against the
# uniform_baseline, at MATCHED training rounds, on:
#   - eval_hard_posthoc : one-step RMSE/NRMSE + loss-profile inequality (Pareto bulk<->hard)
#                         on uniform / hard-tail (tv,lowamp) / off-manifold tube (rho) / coverage sets
#   - eval_rollout_posthoc : per-step rollout NRMSE + stable horizon K_tau (the H1 consequence)
#
# Matched window = completed rounds 10-12 (checkpoint next_round in {11,12,13}). The Phase 2
# jobs were launched with rounds=30 before the budget was cut to 10; A100 seeds ran to round
# 27-30, so we evaluate each run at checkpoint_latest and INCLUDE ONLY the seeds whose latest
# checkpoint sits at round 10-12. This keeps the comparison matched in solver-call budget; the
# round-27-30 outliers and the 4 seed505 runs (no checkpoint) are excluded. The uniform_baseline
# seeds sit at the *high* end (rounds 11-12), so any residual mismatch is conservative for the
# hard-mining variants. Each script prints per-seed rounds so the matching is auditable.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNROOT=${PHASE2_RUNROOT:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_runs/ks800_phase2_5seed}
VDIR=${PHASE2_VDIR:-/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_validation}
UNIFORM=${PHASE2_UNIFORM:-$VDIR/ks_res800_al4pde_sub5_seed43_n1500_t100.npz}
HARD=${PHASE2_HARD:-$VDIR/hard_sub5}
OUTDIR=${PHASE2_OUTDIR:-$VDIR/phase2_round10_analysis}
mkdir -p "$OUTDIR"

VARIANT_SEEDS=(
  uf00_tophalf_seed101 uf00_tophalf_seed303 uf00_tophalf_seed404
  uf25_tophalf_seed101 uf25_tophalf_seed303 uf25_tophalf_seed404
  uf50_ensvar_seed101 uf50_ensvar_seed202 uf50_ensvar_seed303 uf50_ensvar_seed404
  uf50_tophalf_seed202 uf50_tophalf_seed303 uf50_tophalf_seed404
  uf75_tophalf_seed101 uf75_tophalf_seed202 uf75_tophalf_seed303
  uniform_baseline_seed101 uniform_baseline_seed202 uniform_baseline_seed303 uniform_baseline_seed404
)
RUNS=()
for v in "${VARIANT_SEEDS[@]}"; do RUNS+=("$RUNROOT/$v"); done

cd "$ROOT"
echo "==================== eval_hard_posthoc (Pareto bulk<->hard + tube) ===================="
python -u scripts/eval_hard_posthoc.py --runs "${RUNS[@]}" \
  --uniform "$UNIFORM" --hard "$HARD" --baseline uniform_baseline \
  2>&1 | tee "$OUTDIR/hard_posthoc.txt"

echo "==================== eval_rollout_posthoc (K_tau) ===================="
python -u scripts/eval_rollout_posthoc.py --runs "${RUNS[@]}" \
  --uniform "$UNIFORM" --baseline uniform_baseline --steps 100 \
  2>&1 | tee "$OUTDIR/rollout_posthoc.txt"

echo "==================== DONE -> $OUTDIR ===================="
