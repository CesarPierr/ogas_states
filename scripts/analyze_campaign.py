"""5-seed campaign analysis: mean±std per arm + Welch t / Mann-Whitney U vs baselines.

For every arm with >=2 finished seeds under BASE_DIR (<arm>_seed<seed>/history.json,
last entry must be the final round), aggregates the final-round metrics across seeds
and tests each arm against uniform_baseline AND random_tube. At n=5 only large
effects survive — report exact p-values, no stars.

  python scripts/analyze_campaign.py [--base-dir DIR] [--metric nrmse_p50] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

ARMS = ["uniform_baseline", "noise_inject", "random_tube", "mined_ic",
        "gen_v3", "gen_v3_edit", "gen_v3_cfg", "gen_v3_cfg2"]
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
SETS = ["val_hard_tv_div", "val_hard_lowamp_div", "val_hard_mixed_div",
        "val_tube_lowk_rho0p1", "val_tube_lowk_rho0p25", "val_tube_lowk_rho0p5",
        "val_tube_midk_rho0p1", "val_tube_midk_rho0p25", "val_tube_midk_rho0p5",
        "val_tube_highk_rho0p1", "val_tube_highk_rho0p25", "val_tube_highk_rho0p5"]
BULK = ["val/rmse_mean", "val/nrmse_mean", "rollout/nrmse_mean", "rollout/rmse_final_p99"]


def collect(base: Path, arm: str, n_rounds: int) -> dict[str, np.ndarray]:
    """Per-metric arrays across seeds (final round only, complete runs only)."""
    rows = []
    for s in SEEDS:
        f = base / f"{arm}_seed{s}" / "history.json"
        if not f.exists():
            continue
        h = json.load(open(f))
        if not h or h[-1].get("round") != n_rounds - 1:
            continue
        rows.append(h[-1])
    if not rows:
        return {}
    keys = set().union(*rows)
    return {k: np.array([r[k] for r in rows if k in r and np.isfinite(r.get(k, np.nan))])
            for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path,
                    default=Path(os.environ.get("FAST", ".")) / "ogas_states_runs/ks800_v3_pilot")
    ap.add_argument("--metric", default="nrmse_p50")
    ap.add_argument("--n-rounds", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = {a: collect(args.base_dir, a, args.n_rounds) for a in ARMS}
    data = {a: d for a, d in data.items() if d}
    ns = {a: max(len(v) for v in d.values()) for a, d in data.items()}
    print(f"arms found (n seeds): { {a: ns[a] for a in data} }\n")

    metrics = [f"hard_val/{s}/{args.metric}" for s in SETS] + BULK
    report: dict = {"n_seeds": ns, "metric": args.metric, "rows": []}

    for ref_name in ["uniform_baseline", "random_tube"]:
        if ref_name not in data:
            continue
        print(f"===== arm vs {ref_name} (mean±std, % change, Welch p, MWU p) =====")
        for m in metrics:
            short = m.replace("hard_val/", "").replace(f"/{args.metric}", "")
            line = f"{short:<24}"
            ref = data[ref_name].get(m, np.array([]))
            for a in data:
                if a == ref_name or m not in data[a]:
                    continue
                x = data[a][m]
                if len(x) < 2 or len(ref) < 2:
                    continue
                chg = 100 * (x.mean() - ref.mean()) / (abs(ref.mean()) + 1e-12)
                tw = ttest_ind(x, ref, equal_var=False)
                mw = mannwhitneyu(x, ref, alternative="two-sided")
                report["rows"].append({"metric": m, "arm": a, "vs": ref_name,
                                       "mean": float(x.mean()), "std": float(x.std(ddof=1)),
                                       "ref_mean": float(ref.mean()), "ref_std": float(ref.std(ddof=1)),
                                       "pct_change": float(chg), "welch_p": float(tw.pvalue),
                                       "mwu_p": float(mw.pvalue)})
                line += f" | {a[:11]}: {chg:>+7.1f}% (w={tw.pvalue:.3f},u={mw.pvalue:.3f})"
            print(line)
        print()

    out = args.out or (args.base_dir / f"campaign_analysis_{args.metric}.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
