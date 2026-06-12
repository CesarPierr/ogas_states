"""Post-hoc conditioning audit of a pool-based run.

Decomposes the generator's difficulty steering into its mechanisms and measures the
ceiling imposed by label noise. Two parts:

PART 1 (pool npz, CPU, instant) — per round:
  * label stability ceiling: spearman(pretrain difficulty, posttrain difficulty) of the
    SAME states under consecutive surrogates. If a state's difficulty is not stable
    across one training round, no generator conditioned on stale labels can steer
    beyond this correlation.
  * production fidelity: spearman(target_bins, realized difficulty) of generator-sourced
    states (source==1), both at selection time (pretrain labels = same ensemble that
    ranked the candidates) and after training (posttrain labels = what actually mattered).
  * fallback rate (source==2) and generated fraction.

PART 2 (checkpoint, GPU preferred) — final-round generator, three probes through the
real solver (one step each, identical labeling):
  * unselected:  ddpm.sample per target bin (the PURE conditioning signal),
  * selected:    production path generate_states_and_params with the run's
                 candidate_factor + TV gate (conditioning + selection),
  * random_tube: no-learning reference (anchors + low-k noise).
  Reported per probe: spearman(target bin, realized difficulty), monotone fraction and
  dynamic range of per-bin medians, realized-difficulty quantiles, per-bin TV and
  amplitude (is "hard" just "rough" or "low-amp"?).

Usage (GPU job or login CPU with small --n-per-bin):
  python scripts/audit_conditioning.py RUN_DIR [--n-per-bin 128] [--rounds all|last]
      [--skip-checkpoint] [--out audit.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poolbased_surrogate.checkpoint import load_checkpoint
from poolbased_surrogate.config import load_config
from poolbased_surrogate.data import TransitionPool
from poolbased_surrogate.eval import load_validation_data
from poolbased_surrogate.generator_eval import _pde_step_batch
from poolbased_surrogate.models.surrogate import build_surrogate
from poolbased_surrogate.models.ddpm import DDPM1D, FlowMatching1D
from poolbased_surrogate.pde import PDE1D
from poolbased_surrogate.pool import (
    _anchor_states,
    generate_states_and_params,
    make_strategy_transitions,
    state_total_variation,
)
from poolbased_surrogate.train import (
    compute_transition_losses,
    compute_transition_uncertainty,
    realized_difficulty,
)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    from scipy.stats import spearmanr

    r, _ = spearmanr(a[m], b[m])
    return float(r)


# ---------------------------------------------------------------- part 1: pool npz audit
def audit_pools(run_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(run_dir.glob("pool_round_*.npz"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1])):
        rid = int(re.findall(r"\d+", f.stem)[-1])
        z = np.load(f)
        row: dict = {"round": rid, "n": int(len(z["states"]))}
        src = z["source"] if "source" in z else np.zeros(row["n"], dtype=np.int8)
        row["frac_generated"] = float((src == 1).mean())
        row["frac_fallback"] = float((src == 2).mean())

        # Label-stability ceiling (all states; also uniform-only to exclude generator drift).
        for sig, pre_k, post_k in (
            ("uncertainty", "pretrain_uncertainty", "uncertainty"),
            ("loss", "pretrain_losses", "losses"),
        ):
            if pre_k in z and post_k in z and z[pre_k] is not None:
                pre, post = np.asarray(z[pre_k]), np.asarray(z[post_k])
                row[f"stability_{sig}_all"] = spearman(pre, post)
                u = src == 0
                if u.sum() > 10:
                    row[f"stability_{sig}_uniform"] = spearman(pre[u], post[u])

        # Production conditioning+selection fidelity on generated states.
        if "target_bins" in z:
            tb = np.asarray(z["target_bins"])
            g = (src == 1) & (tb >= 0)
            row["n_binned_generated"] = int(g.sum())
            if g.sum() > 10:
                if "pretrain_uncertainty" in z:
                    row["fidelity_at_selection"] = spearman(tb[g], np.asarray(z["pretrain_uncertainty"])[g])
                if "uncertainty" in z:
                    row["fidelity_after_training"] = spearman(tb[g], np.asarray(z["uncertainty"])[g])
                if "pretrain_losses" in z:
                    row["fidelity_loss_at_selection"] = spearman(tb[g], np.asarray(z["pretrain_losses"])[g])
        rows.append(row)
    return rows


# ---------------------------------------------------------- part 2: checkpoint probes
def per_bin_stats(target_bins: np.ndarray, realized: np.ndarray, states: np.ndarray, n_q: int) -> dict:
    """Summary of the bin -> realized-difficulty response curve."""
    med, tvs, amps = [], [], []
    tv = state_total_variation(states)
    amp = np.sqrt((states[:, 0] ** 2).mean(axis=1))
    for k in range(n_q):
        m = (target_bins == k) & np.isfinite(realized)
        med.append(float(np.median(realized[m])) if m.sum() > 2 else float("nan"))
        tvs.append(float(np.median(tv[m])) if m.sum() > 2 else float("nan"))
        amps.append(float(np.median(amp[m])) if m.sum() > 2 else float("nan"))
    med_a = np.asarray(med)
    ok = np.isfinite(med_a)
    diffs = np.diff(med_a[ok])
    iqr = float(np.subtract(*np.nanpercentile(realized[np.isfinite(realized)], [75, 25]))) or 1e-12
    return {
        "spearman_bin_vs_realized": spearman(target_bins.astype(float), realized),
        "monotone_frac": float((diffs > 0).mean()) if len(diffs) else float("nan"),
        "dynamic_range_iqr": float((med_a[ok][-1] - med_a[ok][0]) / iqr) if ok.sum() > 1 else float("nan"),
        "bin_median_realized": med,
        "bin_median_tv": tvs,
        "bin_median_amplitude": amps,
        "realized_q": {q: float(np.nanquantile(realized, q)) for q in (0.1, 0.5, 0.9, 0.99)},
    }


def label_and_score(pde, surrogate, states, params, device, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Solver-step + surrogate scoring; returns (realized difficulty, finite mask)."""
    next_states = _pde_step_batch(pde, states, params, int(cfg.ddpm.solver_batch_size))
    pool = TransitionPool(states=states, params=params, next_states=next_states)
    losses = compute_transition_losses(surrogate, pool, batch_size=512, device=device)
    unc = (
        compute_transition_uncertainty(surrogate, pool, batch_size=512, device=device)
        if getattr(surrogate, "n_models", 1) > 1
        else None
    )
    finite = np.isfinite(states).all(axis=(1, 2)) & np.isfinite(next_states).all(axis=(1, 2)) & np.isfinite(losses)
    real = realized_difficulty(losses, unc, next_states, cfg.ddpm.loss_metric, cfg.ddpm.difficulty_signal)
    return real, finite


def audit_checkpoint(run_dir: Path, cfg, n_per_bin: int, seed: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    pde = PDE1D(cfg.pde)
    validation = load_validation_data(Path(cfg.validation.path))
    state_mean = float(validation.trajectories.mean())
    state_std = float(validation.trajectories.std()) + 1e-8

    model = build_surrogate(
        resolution=cfg.pde.resolution,
        hidden=cfg.surrogate.hidden,
        depth=cfg.surrogate.depth,
        ensemble_size=cfg.surrogate.ensemble_size,
        param_dim=len(pde.param_ranges),
        model_name=cfg.surrogate.model,
        difference_weight=cfg.surrogate.difference_weight,
    ).to(device)
    generator_cls = FlowMatching1D if str(cfg.ddpm.generator).lower() in ("flow", "flow_matching", "fm") else DDPM1D
    ddpm = generator_cls(
        resolution=cfg.pde.resolution,
        hidden=cfg.ddpm.hidden,
        steps=cfg.ddpm.steps,
        loss_conditional=cfg.ddpm.mode == "conditional_loss",
        param_dim=len(pde.param_ranges),
        param_mode=cfg.ddpm.param_mode,
        residual_blocks=cfg.ddpm.residual_blocks,
        kernel_size=cfg.ddpm.kernel_size,
        n_quantiles=cfg.ddpm.n_quantiles if cfg.ddpm.mode == "conditional_quantile" else 0,
        quant_embed_dim=cfg.ddpm.quant_embed_dim,
    ).to(device)
    ckpt = run_dir / "checkpoint_latest.pt"
    rnd, pool, _, _ = load_checkpoint(ckpt, model, ddpm, rng, device)
    model.eval()
    ddpm.eval()
    n_q = int(ddpm.n_quantiles)
    out: dict = {"checkpoint_round": int(rnd) - 1, "n_quantiles": n_q, "device": str(device)}
    src = pool.source if pool.source is not None else np.zeros(len(pool), dtype=np.int8)
    anchors = pool.states[src == 0]
    if len(anchors) == 0:
        anchors = pool.states
    aidx = rng.choice(len(anchors), size=min(4096, len(anchors)), replace=False)
    anchors = np.asarray(anchors[aidx], dtype=np.float32)
    tv_gate = (
        float(cfg.ddpm.realism_tv_gate) * float(np.quantile(state_total_variation(anchors), 0.99))
        if float(cfg.ddpm.realism_tv_gate) > 0
        else 0.0
    )

    with torch.no_grad():
        # --- probe 1: unselected, per-bin (pure conditioning) -----------------------
        tb = np.repeat(np.arange(n_q), n_per_bin)
        params = pde.sample_params_uniform(len(tb), rng)
        pscaled = torch.from_numpy(2.0 * pde.normalize_params(params) - 1.0).float().to(device)
        chunks = []
        for s in range(0, len(tb), int(cfg.ddpm.sample_batch_size)):
            e = min(s + int(cfg.ddpm.sample_batch_size), len(tb))
            st, _ = ddpm.sample(
                e - s,
                loss=None,
                params=pscaled[s:e] if ddpm.use_param_cond else None,
                device=device,
                quantile_label=torch.as_tensor(tb[s:e], dtype=torch.long, device=device),
            )
            chunks.append(st.cpu().numpy())
        states = (np.concatenate(chunks) * state_std + state_mean).astype(np.float32)
        real, fin = label_and_score(pde, model, states, params, device, cfg)
        out["unselected"] = per_bin_stats(tb[fin], real[fin], states[fin], n_q)
        out["unselected"]["finite_ratio"] = float(fin.mean())

        # --- probe 2: production path (conditioning + oversample-reject) ------------
        n_sel = n_per_bin * n_q
        sel_states, sel_params, sel_bins = generate_states_and_params(
            ddpm, n_sel, cfg.ddpm.mode, None, pde, state_mean, state_std, rng, device,
            batch_size=int(cfg.ddpm.sample_batch_size),
            param_prior=cfg.ddpm.param_prior,
            sample_strategy=cfg.ddpm.sample_strategy,
            sample_strategy_temp=cfg.ddpm.sample_strategy_temp,
            surrogate=model,
            candidate_factor=int(cfg.ddpm.candidate_factor),
            tv_gate_threshold=tv_gate,
            anchors_phys=anchors,
            sample_mode=cfg.ddpm.sample_mode,
            edit_t0=float(cfg.ddpm.edit_t0),
        )
        real_s, fin_s = label_and_score(pde, model, sel_states, sel_params, device, cfg)
        ok = fin_s & (sel_bins >= 0)
        out["selected"] = per_bin_stats(sel_bins[ok], real_s[ok], sel_states[ok], n_q)
        out["selected"]["finite_ratio"] = float(fin_s.mean())

        # --- probe 3: random_tube reference (no learning) ---------------------------
        rt_states, rt_params = make_strategy_transitions(
            "random_tube", pde, model, anchors, n_sel, rng, device,
            int(cfg.ddpm.solver_batch_size), float(cfg.pool.tube_rho_min),
            float(cfg.pool.tube_rho_max), int(cfg.pool.tube_kmax), 1,
        )
        real_t, fin_t = label_and_score(pde, model, rt_states, rt_params, device, cfg)
        out["random_tube"] = {
            "realized_q": {q: float(np.nanquantile(real_t[fin_t], q)) for q in (0.1, 0.5, 0.9, 0.99)},
            "finite_ratio": float(fin_t.mean()),
        }
        # Pool/uniform realized-difficulty reference under the SAME (final) surrogate.
        if pool.uncertainty is not None or pool.losses is not None:
            ref = pool.uncertainty if cfg.ddpm.difficulty_signal == "ensemble_var" and pool.uncertainty is not None else pool.losses
            u = src == 0
            out["uniform_pool_reference_q"] = {
                q: float(np.nanquantile(np.asarray(ref)[u], q)) for q in (0.1, 0.5, 0.9, 0.99)
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n-per-bin", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-checkpoint", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.run_dir / "config.resolved.json")
    report: dict = {"run_dir": str(args.run_dir)}
    report["pool_rounds"] = audit_pools(args.run_dir)

    print(f"=== conditioning audit: {args.run_dir.name} ===")
    print(f"{'rnd':>3} {'stab_unc':>9} {'stab_loss':>9} {'fid@sel':>8} {'fid@post':>9} {'fallback':>8}")
    for r in report["pool_rounds"]:
        print(
            f"{r['round']:>3} {r.get('stability_uncertainty_all', float('nan')):>9.3f} "
            f"{r.get('stability_loss_all', float('nan')):>9.3f} "
            f"{r.get('fidelity_at_selection', float('nan')):>8.3f} "
            f"{r.get('fidelity_after_training', float('nan')):>9.3f} "
            f"{r.get('frac_fallback', float('nan')):>8.3f}"
        )

    if not args.skip_checkpoint and (args.run_dir / "checkpoint_latest.pt").exists():
        report["checkpoint"] = audit_checkpoint(args.run_dir, cfg, args.n_per_bin, args.seed)
        for probe in ("unselected", "selected"):
            p = report["checkpoint"][probe]
            print(
                f"{probe:>11}: spearman={p['spearman_bin_vs_realized']:.3f} "
                f"monotone={p['monotone_frac']:.2f} dyn_range={p['dynamic_range_iqr']:.2f} "
                f"realized_med={p['realized_q'][0.5]:.4g} p99={p['realized_q'][0.99]:.4g}"
            )
        rt = report["checkpoint"]["random_tube"]
        print(f"{'random_tube':>11}: realized_med={rt['realized_q'][0.5]:.4g} p99={rt['realized_q'][0.99]:.4g}")

    out_path = args.out or (args.run_dir / "conditioning_audit.json")
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
