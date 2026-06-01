#!/usr/bin/env python
"""Analyze a loss-generator sweep and print recommendations.

Usage:
    python scripts/analyze_sweep.py <sweep_output_dir>
    python scripts/analyze_sweep.py <sweep_output_dir> --recommend-json out.json

Reads score_metrics.json from each sub-directory and emits:
  - Comparison table (TV, loss ratio, rank correlation per strategy)
  - Best model recommendation for integration into the main AL loop
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


STRATEGIES = ["top1", "top5", "top_half", "exp_bias", "uniform"]

# New comprehensive metrics added in ablation_v2
NEW_METRICS = [
    "psd_log_l2",          # spectral realism (0 = perfect)
    "amplitude_ratio",     # amplitude match (1 = perfect)
    "calibration_mae",     # conditioning accuracy (0 = perfect)
    "calibration_monotone_frac",  # monotone conditioning (1 = perfect)
    "conditioning_mi_nats",       # mutual information (log(nq) = perfect)
    "intra_bin_diversity_mean",   # within-bin diversity
    "gen_time_per_sample_ms",     # generation speed
]


def load_sweep(sweep_dir: Path) -> dict[str, dict]:
    results = {}
    for sub in sorted(sweep_dir.iterdir()):
        score_path = sub / "score_metrics.json"
        hist_path = sub / "history.json"
        eval_hist_path = sub / "eval_history.json"
        if not score_path.exists():
            print(f"  [skip] {sub.name}: no score_metrics.json")
            continue
        with open(score_path) as f:
            score = json.load(f)
        last_loss = float("nan")
        epoch_time = float("nan")
        total_train_time = float("nan")
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            if hist:
                last_loss = float(hist[-1].get("train/loss", float("nan")))
                epoch_times = [float(h.get("epoch_time_s", float("nan"))) for h in hist]
                finite_et = [e for e in epoch_times if math.isfinite(e)]
                if finite_et:
                    epoch_time = sum(finite_et) / len(finite_et)
                total_train_time = float(hist[-1].get("wall_time_s", float("nan")))
        eval_history: list[dict] = []
        if eval_hist_path.exists():
            with open(eval_hist_path) as f:
                eval_history = json.load(f)
        results[sub.name] = {
            "score": score,
            "train_loss": last_loss,
            "epoch_time_s": epoch_time,
            "total_train_time_s": total_train_time,
            "eval_history": eval_history,
        }
    return results


def best_strategy(score: dict, strategies: list[str]) -> tuple[str, float]:
    """Return (strategy_name, loss_ratio) for the strategy with highest loss ratio."""
    best_name, best_ratio = "none", -1.0
    for s in strategies:
        ratio = score.get(f"{s}/loss_vs_reference_ratio", float("nan"))
        if math.isfinite(ratio) and ratio > best_ratio:
            best_ratio = ratio
            best_name = s
    return best_name, best_ratio


def print_table(results: dict[str, dict]) -> None:
    ref_tv = results[next(iter(results))]["score"].get("validation_states/total_variation_mean",
        results[next(iter(results))]["score"].get("reference_states/total_variation_mean", float("nan")))
    header = f"{'Model':<22} {'params':>7} {'steps':>5} {'TV gen':>7} {'TV ratio':>8}"
    for s in STRATEGIES:
        header += f" {s[:7]+'/r':>9}"
    header += f" {'best_strat':>12} {'train_loss':>10} {'epoch_s':>7} {'tot_t_s':>8} {'best_ep':>7} {'conv_ev':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    print(f"  Real KS TV = {ref_tv:.1f}")
    print("-" * len(header))
    for name, data in results.items():
        sc = data["score"]
        n_params = sc.get("model/n_params", 0)
        steps = int(sc.get("model/steps", 0))
        tv = sc.get("top1/states/total_variation_mean", float("nan"))
        tv_ratio = tv / ref_tv if math.isfinite(tv) and ref_tv > 0 else float("nan")
        best_s, best_r = best_strategy(sc, STRATEGIES)
        row = f"{name:<22} {n_params:>7.0f} {steps:>5d} {tv:>7.1f} {tv_ratio:>8.2f}x"
        for s in STRATEGIES:
            ratio = sc.get(f"{s}/loss_vs_reference_ratio", float("nan"))
            row += f" {ratio:>9.2f}"
        # cost columns
        best_ep = best_eval_epoch_from_history(data) if data.get("eval_history") else float("nan")
        conv_ev = convergence_eval_epoch_from_history(data) if data.get("eval_history") else float("nan")
        row += (
            f" {best_s:>12} {data['train_loss']:>10.4f}"
            f" {data.get('epoch_time_s', float('nan')):>7.2f}"
            f" {data.get('total_train_time_s', float('nan')):>8.1f}"
            f" {best_ep:>7.0f}"
            f" {conv_ev:>7.0f}"
        )
        print(row)
    print("=" * len(header))


def best_eval_epoch_from_history(data: dict) -> float:
    eh = data.get("eval_history") or []
    if not eh:
        return float("nan")
    pairs = [(float(e.get("epoch", float("nan"))), float(e.get("combined_score", float("nan")))) for e in eh]
    pairs = [(ep, s) for ep, s in pairs if math.isfinite(ep) and math.isfinite(s)]
    if not pairs:
        return float("nan")
    return max(pairs, key=lambda x: x[1])[0]


def convergence_eval_epoch_from_history(data: dict, frac: float = 0.95) -> float:
    eh = data.get("eval_history") or []
    if not eh:
        return float("nan")
    pairs = [(float(e.get("epoch", float("nan"))), float(e.get("combined_score", float("nan")))) for e in eh]
    pairs = [(ep, s) for ep, s in pairs if math.isfinite(ep) and math.isfinite(s)]
    if not pairs:
        return float("nan")
    best = max(s for _, s in pairs)
    if best <= 0:
        return min(ep for ep, _ in pairs)
    threshold = frac * best
    for ep, s in sorted(pairs, key=lambda x: x[0]):
        if s >= threshold:
            return ep
    return max(ep for ep, _ in pairs)


def print_eval_curves(results: dict[str, dict]) -> None:
    """Print compact per-config epoch curve: combined_score vs epoch, plus train time."""
    if not any(d.get("eval_history") for d in results.values()):
        return
    print("\n[per-config validation curve over epochs — combined_score]")
    epochs_seen = sorted({int(e["epoch"]) for d in results.values() for e in (d.get("eval_history") or [])})
    hdr = f"{'Model':<22} " + " ".join(f"ep{ep:>3}" for ep in epochs_seen) + f" {'best_ep':>7} {'epoch_s':>7} {'tot_t_s':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, data in results.items():
        eh = data.get("eval_history") or []
        by_ep = {int(e["epoch"]): float(e.get("combined_score", float("nan"))) for e in eh}
        row = f"{name:<22} " + " ".join(f"{by_ep.get(ep, float('nan')):>5.2f}" for ep in epochs_seen)
        row += f" {best_eval_epoch_from_history(data):>7.0f} {data.get('epoch_time_s', float('nan')):>7.2f} {data.get('total_train_time_s', float('nan')):>8.1f}"
        print(row)


def recommend(results: dict[str, dict], tv_ratio_max: float = 3.0) -> dict:
    """Pick the best model for integration into the main AL loop.

    Hard constraint: TV ratio must be <= tv_ratio_max (default 3×).
    States with TV > 3× real KS are too rough to reliably survive PDE stepping
    and inflate loss ratios for the wrong reason (solver divergence artefacts).

    Among valid models, we score on:
      - TV closeness to real KS:  score = exp(-0.5 * (tv_ratio - 1)^2)
      - Loss ratio under best strategy (gen states harder than reference pool)
      - Rank correlation (quantile conditioning actually controls difficulty)

    Combined = 0.4 * tv_score + 0.4 * loss_ratio_norm + 0.2 * rank_corr_norm
    """
    if not results:
        return {}
    ref_tv = results[next(iter(results))]["score"].get("reference_states/total_variation_mean", 24.0)

    # --- filter to physically valid models ---
    valid = {}
    invalid = {}
    for name, data in results.items():
        sc = data["score"]
        tv = sc.get("top1/states/total_variation_mean", float("nan"))
        tv_ratio = tv / ref_tv if math.isfinite(tv) and ref_tv > 0 else float("nan")
        if math.isfinite(tv_ratio) and tv_ratio <= tv_ratio_max:
            valid[name] = data
        else:
            invalid[name] = (tv_ratio, tv)

    if invalid:
        print(f"\n[excluded: TV ratio > {tv_ratio_max}×]")
        for n, (r, tv) in invalid.items():
            print(f"  {n}: TV={tv:.1f} ({r:.1f}×)")

    if not valid:
        print("No models pass the TV ratio filter — relaxing to best available.")
        valid = results

    all_ratios, all_corrs = [], []
    for data in valid.values():
        sc = data["score"]
        _, best_r = best_strategy(sc, STRATEGIES)
        if math.isfinite(best_r):
            all_ratios.append(best_r)
        for s in STRATEGIES:
            c = sc.get(f"{s}/target_vs_realized_rank_corr", float("nan"))
            if math.isfinite(c):
                all_corrs.append(c)
    # Clamp loss ratio at 5 to avoid artefact-inflated values dominating
    max_ratio = min(max(all_ratios, default=1.0), 5.0)
    min_corr = min(all_corrs, default=0.0)
    max_corr = max(all_corrs, default=1.0)

    # Cost normalization: epoch_time_s × convergence_epoch_eval (or fall back to full train).
    train_costs = []
    for d in valid.values():
        et = d.get("epoch_time_s", float("nan"))
        conv = convergence_eval_epoch_from_history(d) if d.get("eval_history") else float("nan")
        if math.isfinite(et) and math.isfinite(conv):
            train_costs.append(et * conv)
        elif math.isfinite(d.get("total_train_time_s", float("nan"))):
            train_costs.append(d["total_train_time_s"])
    min_cost = min(train_costs, default=1.0) or 1.0
    max_cost = max(train_costs, default=1.0) or 1.0

    scored = {}
    for name, data in valid.items():
        sc = data["score"]
        tv = sc.get("top1/states/total_variation_mean", float("nan"))
        tv_ratio = tv / ref_tv if math.isfinite(tv) and ref_tv > 0 else float("nan")
        import math as _m
        tv_score = _m.exp(-0.5 * (tv_ratio - 1.0) ** 2) if math.isfinite(tv_ratio) else 0.0
        _, best_r = best_strategy(sc, STRATEGIES)
        ratio_norm = min((best_r / max_ratio), 1.0) if math.isfinite(best_r) and max_ratio > 0 else 0.0
        best_corr = max(
            (sc.get(f"{s}/target_vs_realized_rank_corr", float("nan")) for s in STRATEGIES),
            default=float("nan"),
            key=lambda x: x if math.isfinite(x) else -999.0,
        )
        corr_norm = ((best_corr - min_corr) / (max_corr - min_corr + 1e-8)) if math.isfinite(best_corr) else 0.0
        quality = 0.4 * tv_score + 0.4 * ratio_norm + 0.2 * corr_norm

        # Training cost: epoch_time_s × eval-convergence epoch, normalized to [0,1] (lower=cheaper).
        et = data.get("epoch_time_s", float("nan"))
        conv_ev = convergence_eval_epoch_from_history(data) if data.get("eval_history") else float("nan")
        if math.isfinite(et) and math.isfinite(conv_ev):
            train_cost = et * conv_ev
        else:
            train_cost = data.get("total_train_time_s", float("nan"))
        if math.isfinite(train_cost) and max_cost > min_cost:
            cost_norm = (train_cost - min_cost) / (max_cost - min_cost)  # 0=cheapest, 1=slowest
        else:
            cost_norm = 0.0
        cost_score = 1.0 - cost_norm
        # Final combined: 70% quality, 30% training speed reward.
        combined = 0.7 * quality + 0.3 * cost_score

        scored[name] = {
            "combined_score": combined,
            "quality_score": quality,
            "cost_score": cost_score,
            "tv_ratio": tv_ratio,
            "tv_score": tv_score,
            "best_loss_ratio": best_r if math.isfinite(best_r) else None,
            "best_rank_corr": best_corr if math.isfinite(best_corr) else None,
            "steps": int(sc.get("model/steps", 0)),
            "hidden": int(sc.get("model/hidden", 0)),
            "residual_blocks": int(sc.get("model/residual_blocks", 0)),
            "n_params": int(sc.get("model/n_params", 0)),
            "epoch_time_s": data.get("epoch_time_s"),
            "convergence_epoch_eval": conv_ev if math.isfinite(conv_ev) else None,
            "best_eval_epoch": best_eval_epoch_from_history(data) if data.get("eval_history") else None,
            "train_cost_s": train_cost,
            "train_loss": data["train_loss"],
        }

    if not scored:
        return {}

    best_name = max(scored, key=lambda k: scored[k]["combined_score"])
    rec = {"best_model": best_name, **scored[best_name], "all_models": scored}
    return rec


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", help="Sweep output directory")
    parser.add_argument("--recommend-json", default=None, help="Write recommendation JSON to this path")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        help="Strategies to show in table")
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    print(f"\nAnalyzing sweep: {sweep_dir}")
    results = load_sweep(sweep_dir)
    if not results:
        print("No completed runs found.")
        return

    print_table(results)
    print_eval_curves(results)
    rec = recommend(results)
    if rec:
        print(f"\n{'='*60}")
        print(f"RECOMMENDATION: {rec['best_model']}")
        print(f"  steps={rec['steps']}  hidden={rec['hidden']}  rb={rec['residual_blocks']}")
        print(f"  TV ratio={rec['tv_ratio']:.2f}x (1.0 = perfect)")
        print(f"  best loss ratio={rec['best_loss_ratio']}")
        print(f"  best rank corr={rec['best_rank_corr']}")
        if rec.get('convergence_epoch_eval') is not None:
            print(f"  convergence epoch (eval)={rec['convergence_epoch_eval']:.0f}  "
                  f"epoch_time={rec.get('epoch_time_s', float('nan')):.2f}s  "
                  f"→ train_cost≈{rec['train_cost_s']:.0f}s")
        print(f"  quality_score={rec.get('quality_score', float('nan')):.3f}  "
              f"cost_score={rec.get('cost_score', float('nan')):.3f}  "
              f"combined={rec['combined_score']:.3f}")
        print(f"{'='*60}")
        if args.recommend_json:
            Path(args.recommend_json).write_text(json.dumps(rec, indent=2))
            print(f"Recommendation written to {args.recommend_json}")


if __name__ == "__main__":
    main()
