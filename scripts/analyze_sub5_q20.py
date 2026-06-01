#!/usr/bin/env python
"""Per-seed analysis of the sub5 q20 production experiment.

For each variant x seed, reads history.json (per-round metrics) and extracts the
final-round surrogate quality and generator diagnostics. Aggregates across seeds
(mean +/- std) and compares each generator variant against the uniform baseline.

Usage:
    python scripts/analyze_sub5_q20.py <base_dir> [--variants v1 v2 ...] [--baseline uniform]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SEEDS = [101, 202, 303, 404, 505]
VARIANTS = [
    "expbias", "expbias_t03", "tophalf", "top3", "uniform",
]
# Final-round metrics we care about. Lower is better for all of these.
METRICS = ["val/nrmse_mean", "val/nrmse_p50", "val/nrmse_p95", "val/nrmse_p99",
           "rollout/nrmse_mean", "rollout/nrmse_final"]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return m, sd


def load_final(base: Path, variant: str, seed: int) -> dict | None:
    hist_path = base / f"{variant}_seed{seed}" / "history.json"
    if not hist_path.exists():
        return None
    try:
        hist = json.loads(hist_path.read_text())
    except Exception:
        return None
    if not hist:
        return None
    last = hist[-1]
    out = {"_n_rounds": len(hist), "_round": last.get("round", len(hist) - 1)}
    for m in METRICS:
        out[m] = last.get(m)
    # generator diagnostics (only present for non-uniform variants)
    gen = last.get("pool/generated_posttrain_loss_mean")
    uni = last.get("pool/uniform_posttrain_loss_mean")
    out["hardness_ratio"] = (gen / uni) if (gen and uni) else None
    out["gen_TV"] = last.get("pool/generated_states_total_variation_mean")
    out["gen_finite"] = last.get("pool/generated_states_finite_ratio")
    out["gen_diversity"] = last.get("pool/generated_pairwise_l2_mean")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--baseline", default="uniform")
    args = ap.parse_args(argv)
    base = Path(args.base_dir)

    # Gather per-variant per-seed
    data: dict[str, dict[int, dict]] = {}
    for v in args.variants:
        data[v] = {}
        for s in args.seeds:
            r = load_final(base, v, s)
            if r is not None:
                data[v][s] = r

    # Completion table
    print("\n=== COMPLETION (rounds reached per seed) ===")
    hdr = f"{'variant':<20}" + "".join(f"s{s:>5}" for s in args.seeds) + f"{'done':>6}"
    print(hdr)
    print("-" * len(hdr))
    for v in args.variants:
        row = f"{v:<20}"
        done = 0
        for s in args.seeds:
            n = data[v].get(s, {}).get("_n_rounds")
            row += f"{(n if n is not None else 0):>6}"
            if n and n >= 20:
                done += 1
        print(row + f"{done:>6}")

    # Surrogate quality table (mean +/- std across seeds)
    print("\n=== SURROGATE QUALITY @ final round (mean +/- std over seeds; LOWER=better) ===")
    hdr = f"{'variant':<20}" + "".join(f"{m.split('/')[-1]:>16}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    baseline_means = {}
    for v in args.variants:
        row = f"{v:<20}"
        for m in METRICS:
            mean, sd = _mean_std([data[v][s].get(m) for s in data[v]])
            if v == args.baseline:
                baseline_means[m] = mean
            row += f"{mean:>9.4f}±{sd:<5.4f}"[:16]
        print(row)

    # Improvement vs baseline (negative = better than uniform)
    if args.baseline in data and data[args.baseline]:
        print(f"\n=== % CHANGE vs '{args.baseline}' baseline (negative = better surrogate) ===")
        hdr = f"{'variant':<20}" + "".join(f"{m.split('/')[-1]:>16}" for m in METRICS)
        print(hdr)
        print("-" * len(hdr))
        for v in args.variants:
            if v == args.baseline:
                continue
            row = f"{v:<20}"
            for m in METRICS:
                mean, _ = _mean_std([data[v][s].get(m) for s in data[v]])
                bm = baseline_means.get(m)
                pct = (mean - bm) / bm * 100 if bm and math.isfinite(mean) and bm != 0 else float("nan")
                row += f"{pct:>+15.1f}%"
            print(row)

    # Generator diagnostics
    print("\n=== GENERATOR DIAGNOSTICS @ final round (mean over seeds) ===")
    hdr = f"{'variant':<20}{'hardness_ratio':>16}{'gen_TV':>10}{'gen_finite':>12}{'gen_diversity':>14}"
    print(hdr)
    print("-" * len(hdr))
    for v in args.variants:
        if v == args.baseline:
            continue
        hr, _ = _mean_std([data[v][s].get("hardness_ratio") for s in data[v]])
        tv, _ = _mean_std([data[v][s].get("gen_TV") for s in data[v]])
        fr, _ = _mean_std([data[v][s].get("gen_finite") for s in data[v]])
        dv, _ = _mean_std([data[v][s].get("gen_diversity") for s in data[v]])
        print(f"{v:<20}{hr:>16.2f}{tv:>10.1f}{fr:>12.3f}{dv:>14.2f}")
    print("\nhardness_ratio = mean surrogate loss on generated states / on uniform states (>1 = harder).")
    print("Real KS validation TV ~ 24.7 (gen_TV close to that = realistic).")


if __name__ == "__main__":
    main()
