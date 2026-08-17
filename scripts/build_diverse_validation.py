#!/usr/bin/env python
"""Build a DIVERSE (not just harder) hard/tube validation suite from a uniform bank.

Fixes three defects of build_hard_validation.py / build_tube_validation.py:
 1. temporal concentration: top-q hard sets collapse onto a handful of source
    trajectories of temporally correlated consecutive frames -> enforce a max number
    of frames per source trajectory and a minimum time gap between selected frames;
 2. single perturbation family: only low-wavenumber (k<=12, 1/k spectrum) tube
    directions were probed -> add mid (13..60) and high (61..200) wavenumber bands;
 3. parameter concentration: hard states cluster in a corner of (param0, param1) ->
    param-stratified selection over a 4x4 quantile grid of the bank's params.

Outputs npz files with the uniform-validation schema (states0, params,
trajectories=[state, next_state]) so they load via eval.load_hard_validation_sets
(validation.hard_dir), plus a manifest.json with per-set diagnostics.

Sets:
  A. diverse hard (selected from bank transitions u_t -> u_{t+1}, t>=1):
     val_hard_tv_div.npz       top-5% input-state total variation, diversified
     val_hard_lowamp_div.npz   smallest-5% target-state RMS, diversified
     val_hard_mixed_div.npz    80-95th pct TV band (moderate-hard), diversified
  B. multi-band tube (attractor frames + band-limited perturbation, 1 solver step):
     val_tube_{low,mid,high}k_rho{0p1,0p25,0p5}.npz   (9 sets)

Usage:
    python scripts/build_diverse_validation.py --bank <uniform.npz> \
        --config <config.(yaml|resolved.json)> --output-dir <dir> [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from poolbased_surrogate.config import PDEConfig, load_config
from poolbased_surrogate.pde import PDE1D

TUBE_BANDS = {"lowk": (1, 12), "midk": (13, 60), "highk": (61, 200)}


def load_pde_cfg(path: str) -> PDEConfig:
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())["pde"]
        return PDEConfig(**d)
    return load_config(path).pde


def total_variation(states: np.ndarray) -> np.ndarray:
    """Model-independent difficulty: periodic total variation per state. states: (N,...,L)."""
    s = states.reshape(states.shape[0], -1).astype(np.float64)
    tv = np.abs(np.diff(s, axis=-1)).sum(-1)
    tv += np.abs(s[:, 0] - s[:, -1])  # periodic wrap
    return tv


def rms(states: np.ndarray) -> np.ndarray:
    s = states.reshape(states.shape[0], -1).astype(np.float64)
    return np.sqrt(np.mean(s**2, axis=-1))


def band_perturbation(n: int, L: int, k1: int, k2: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS, mean-zero real fields with a 1/k spectrum restricted to wavenumbers
    k1..k2 (periodic). Mirrors build_tube_validation.smooth_perturbation but band-limited."""
    k2 = min(k2, L // 2 - 1)
    ks = np.arange(k1, k2 + 1)
    x = np.linspace(0, 2 * np.pi, L, endpoint=False)
    cos = np.cos(np.outer(ks, x))  # (nk, L)
    sin = np.sin(np.outer(ks, x))
    a = rng.normal(size=(n, len(ks))) / ks[None, :]
    b = rng.normal(size=(n, len(ks))) / ks[None, :]
    field = a @ cos + b @ sin
    field -= field.mean(axis=1, keepdims=True)
    field /= np.sqrt(np.mean(field**2, axis=1, keepdims=True)) + 1e-12
    return field.astype(np.float32)


def param_cells(params: np.ndarray, edges0: np.ndarray, edges1: np.ndarray, grid: int) -> np.ndarray:
    """Flat grid x grid quantile-cell id per row of params, from (param0, param1).

    Single-parameter PDEs (e.g. 1D Burgers, params.shape[1] == 1) have no second
    parameter axis: edges1 is then a degenerate constant grid and every row maps to
    column 0 of the second axis, so stratification is purely over param0."""
    c0 = np.clip(np.digitize(params[:, 0], edges0[1:-1]), 0, grid - 1)
    if params.shape[1] < 2:
        c1 = np.zeros(len(params), dtype=np.int64)
    else:
        c1 = np.clip(np.digitize(params[:, 1], edges1[1:-1]), 0, grid - 1)
    return c0 * grid + c1


def diversity_filter(
    cand: np.ndarray,
    priority: np.ndarray,
    src_traj: np.ndarray,
    src_time: np.ndarray,
    max_per_traj: int,
    min_gap: int,
) -> np.ndarray:
    """Greedy accept candidates in decreasing priority, keeping at most max_per_traj
    frames per source trajectory and a >= min_gap time gap within a trajectory."""
    order = cand[np.argsort(-priority[cand], kind="stable")]
    taken: dict[int, list[int]] = defaultdict(list)
    out = []
    for idx in order:
        i, t = int(src_traj[idx]), int(src_time[idx])
        ts = taken[i]
        if len(ts) >= max_per_traj or any(abs(t - t2) < min_gap for t2 in ts):
            continue
        ts.append(t)
        out.append(idx)
    return np.asarray(out, dtype=np.int64)


def param_stratify(
    sel: np.ndarray,
    cell_of: np.ndarray,
    target: int,
    n_cells: int,
    rng: np.random.Generator,
    priority: np.ndarray | None = None,
) -> np.ndarray:
    """Equal quota per non-empty param cell (target total ~target). Within a cell pick by
    priority (most extreme first) or uniformly at random when priority is None."""
    cells = cell_of[sel] if len(sel) else np.empty(0, dtype=np.int64)
    nonempty = np.unique(cells)
    if len(nonempty) == 0:
        return np.empty(0, dtype=np.int64)
    quota = int(np.ceil(target / len(nonempty)))
    out = []
    for c in nonempty:
        members = sel[cells == c]
        if priority is not None:
            members = members[np.argsort(-priority[members], kind="stable")][:quota]
        else:
            members = rng.choice(members, size=min(quota, len(members)), replace=False)
        out.append(members)
    return np.sort(np.concatenate(out))


def save_set(path: Path, state: np.ndarray, nxt: np.ndarray, par: np.ndarray) -> None:
    """Same schema as the uniform validation set: length-2 trajectories [state, next]."""
    s0 = state.reshape(len(state), 1, -1).astype(np.float32)
    traj = np.stack([s0[:, 0], nxt.reshape(len(nxt), -1).astype(np.float32)], axis=1)
    traj = traj[:, :, None, :].astype(np.float32)  # (N,2,1,L)
    np.savez_compressed(path, states0=s0, params=par.astype(np.float32), trajectories=traj)
    print(f"  wrote {path}  ({len(state)} transitions)")


def set_stats(
    state: np.ndarray,
    nxt: np.ndarray,
    par: np.ndarray,
    src: np.ndarray,
    edges0: np.ndarray,
    edges1: np.ndarray,
    grid: int,
) -> dict:
    tv = total_variation(state)
    occ = np.zeros((grid, grid), dtype=int)
    cells = param_cells(par, edges0, edges1, grid)
    for c in cells:
        occ[c // grid, c % grid] += 1
    return {
        "n_transitions": int(len(state)),
        "n_unique_source_trajectories": int(len(np.unique(src))),
        "param_cell_occupancy": occ.tolist(),
        "tv_mean": float(tv.mean()) if len(tv) else None,
        "tv_std": float(tv.std()) if len(tv) else None,
        "rms_in_mean": float(rms(state).mean()) if len(state) else None,
        "rms_target_mean": float(rms(nxt).mean()) if len(nxt) else None,
    }


def solve_one_step(pde: PDE1D, states: np.ndarray, params: np.ndarray, batch: int, tag: str) -> np.ndarray:
    """Batched single solver step (no warmup). Pads the final batch to a fixed shape so
    the JAX solver compiles once."""
    n = len(states)
    out = np.empty_like(states)
    n_batches = (n + batch - 1) // batch
    t0 = time.time()
    for bi, i in enumerate(range(0, n, batch)):
        j = min(i + batch, n)
        s, p = states[i:j], params[i:j]
        if j - i < batch:  # pad to the compiled batch shape
            pad = batch - (j - i)
            s = np.concatenate([s, np.repeat(s[-1:], pad, axis=0)])
            p = np.concatenate([p, np.repeat(p[-1:], pad, axis=0)])
        out[i:j] = pde.step(s, p)[: j - i]
        print(f"    {tag}: batch {bi + 1}/{n_batches} ({time.time() - t0:.1f}s)", flush=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, help="uniform validation bank npz (trajectories [N,T+1,1,X], params [N,P])")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", required=True, help="experiment yaml (or resolved.json) for PDE construction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=int, default=1500, help="target size of each hard set")
    ap.add_argument("--tube-n", type=int, default=1500, help="target size of each tube set")
    ap.add_argument("--tube-n-reduced", type=int, default=1000, help="fallback tube size when the budget is exceeded")
    ap.add_argument("--tube-budget-min", type=float, default=40.0, help="solver-time budget (minutes) for all tube sets")
    ap.add_argument("--solver-batch", type=int, default=256)
    ap.add_argument("--grid", type=int, default=4, help="param-quantile grid is grid x grid")
    ap.add_argument("--max-per-traj", type=int, default=3, help="max hard frames per source trajectory")
    ap.add_argument("--min-gap", type=int, default=10, help="min time gap between hard frames of the same trajectory")
    ap.add_argument("--tube-max-per-traj", type=int, default=2, help="max tube base frames per source trajectory")
    ap.add_argument("--rho", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    with np.load(args.bank) as d:
        traj, params = d["trajectories"], d["params"]
    n_traj, tp1, _, L = traj.shape
    T = tp1 - 1
    frames = traj[:, :, 0, :]  # (n, T+1, L)
    print(f"bank: {n_traj} trajectories x {T} steps (L={L}, P={params.shape[1]})")

    # 4x4 quantile grid over (param0, param1) of the bank
    grid = args.grid
    edges0 = np.quantile(params[:, 0], np.linspace(0, 1, grid + 1))
    if params.shape[1] >= 2:
        edges1 = np.quantile(params[:, 1], np.linspace(0, 1, grid + 1))
    else:  # single-param PDE (e.g. Burgers): degenerate second axis, one column
        edges1 = np.zeros(grid + 1)
    traj_cell = param_cells(params, edges0, edges1, grid)

    # ---- per-transition difficulty, skipping t=0 (transitions u_t -> u_{t+1}, t=1..T-1) ----
    tv_in = np.empty((n_traj, T - 1))
    rms_tgt = np.empty((n_traj, T - 1))
    for i in range(0, n_traj, 100):  # chunked: keep float64 temporaries small
        j = min(i + 100, n_traj)
        blk = frames[i:j]
        tv_in[i:j] = total_variation(blk[:, 1:T].reshape(-1, L)).reshape(j - i, T - 1)
        rms_tgt[i:j] = rms(blk[:, 2:].reshape(-1, L)).reshape(j - i, T - 1)
    tv_flat = tv_in.ravel()
    rms_flat = rms_tgt.ravel()
    src_traj = np.repeat(np.arange(n_traj), T - 1)
    src_time = np.tile(np.arange(1, T), n_traj)
    cell_flat = traj_cell[src_traj]
    n_trans = len(tv_flat)
    print(f"transitions: {n_trans}; TV med={np.median(tv_flat):.1f} target-RMS med={np.median(rms_flat):.3f}")

    manifest: dict = {
        "bank": str(args.bank),
        "config": str(args.config),
        "seed": args.seed,
        "grid": grid,
        "param_edges": {"param0": edges0.tolist(), "param1": edges1.tolist()},
        "hard_constraints": {"max_per_traj": args.max_per_traj, "min_gap": args.min_gap},
        "tube_bands": {k: list(v) for k, v in TUBE_BANDS.items()},
        "rho": args.rho,
        "notes": [],
        "sets": {},
    }

    # ---------------- Set A: diverse hard ----------------
    def build_hard(name: str, cand: np.ndarray, priority: np.ndarray, criterion: str, random_in_cell: bool) -> None:
        div = diversity_filter(cand, priority, src_traj, src_time, args.max_per_traj, args.min_gap)
        sel = param_stratify(div, cell_flat, args.target, grid * grid, rng,
                             priority=None if random_in_cell else priority)
        i, t = src_traj[sel], src_time[sel]
        state = frames[i, t][:, None, :].astype(np.float32)       # (M,1,L)
        nxt = frames[i, t + 1][:, None, :].astype(np.float32)
        par = params[i].astype(np.float32)
        finite = np.isfinite(state).all(axis=(1, 2)) & np.isfinite(nxt).all(axis=(1, 2))
        n_drop = int((~finite).sum())
        state, nxt, par, i = state[finite], nxt[finite], par[finite], i[finite]
        save_set(out / f"{name}.npz", state, nxt, par)
        entry = set_stats(state, nxt, par, i, edges0, edges1, grid)
        entry.update({"criterion": criterion, "n_candidates": int(len(cand)),
                      "n_after_diversity": int(len(div)), "n_nonfinite_dropped": n_drop})
        manifest["sets"][name] = entry
        print(f"  {name}: cand={len(cand)} -> diversity={len(div)} -> stratified={len(state)} "
              f"({entry['n_unique_source_trajectories']} unique trajectories)")

    # ---------------- Original sets + p05, p10, p15 variants ----------------
    for p in [5, 10, 15]:
        thr_tv = np.quantile(tv_flat, 1.0 - p/100.0)
        build_hard(f"val_hard_tv_div_p{p:02d}", np.where(tv_flat >= thr_tv)[0], tv_flat,
                   f"top-{p}% input-state total variation", random_in_cell=False)
    # Alias for backward compatibility
    thr_tv = np.quantile(tv_flat, 0.95)
    build_hard("val_hard_tv_div", np.where(tv_flat >= thr_tv)[0], tv_flat,
               "top-5% input-state total variation", random_in_cell=False)

    for p in [5, 10, 15]:
        thr_amp = np.quantile(rms_flat, p/100.0)
        build_hard(f"val_hard_lowamp_div_p{p:02d}", np.where(rms_flat <= thr_amp)[0], -rms_flat,
                   f"smallest-{p}% target-state RMS", random_in_cell=False)
    # Alias for backward compatibility
    thr_amp = np.quantile(rms_flat, 0.05)
    build_hard("val_hard_lowamp_div", np.where(rms_flat <= thr_amp)[0], -rms_flat,
               "smallest-5% target-state RMS", random_in_cell=False)

    lo, hi = np.quantile(tv_flat, [0.80, 0.95])
    mixed_cand = np.where((tv_flat >= lo) & (tv_flat < hi))[0]
    build_hard("val_hard_mixed_div", mixed_cand, rng.random(n_trans),
               "80-95th percentile band of input-state TV (moderate-hard)", random_in_cell=True)

    # ---------------- Set B: multi-band tube ----------------
    pde = PDE1D(load_pde_cfg(args.config))

    # probe solver speed (compile + steady-state) to keep within the time budget
    probe = frames[: args.solver_batch, T // 2].astype(np.float32)
    probe_p = params[: args.solver_batch].astype(np.float32)
    t0 = time.time()
    pde.step(probe, probe_p)
    t_compile = time.time() - t0
    t0 = time.time()
    pde.step(probe, probe_p)
    t_batch = time.time() - t0
    n_sets = len(TUBE_BANDS) * len(args.rho)
    tube_n = args.tube_n

    def est_minutes(n: int) -> float:
        return (t_compile + n_sets * np.ceil(n / args.solver_batch) * t_batch) / 60.0

    print(f"solver probe: compile {t_compile:.1f}s, batch({args.solver_batch}) {t_batch:.1f}s "
          f"-> est {est_minutes(tube_n):.1f} min for {n_sets} tube sets of {tube_n}")
    if est_minutes(tube_n) > args.tube_budget_min:
        tube_n = args.tube_n_reduced
        msg = (f"tube sets reduced from {args.tube_n} to {tube_n} samples each: projected solver "
               f"time exceeded the {args.tube_budget_min:.0f} min budget "
               f"(est {est_minutes(args.tube_n):.1f} min at n={args.tube_n})")
        manifest["notes"].append(msg)
        print(f"  NOTE: {msg}")
    manifest["tube_n_per_set"] = tube_n

    # base states: late-half attractor frames, max 2 per trajectory, param-stratified
    late_t = np.arange(tp1 // 2, tp1)
    base_src = np.repeat(np.arange(n_traj), len(late_t))
    base_time = np.tile(late_t, n_traj)
    base_idx = np.arange(len(base_src))
    base_sel = diversity_filter(base_idx, rng.random(len(base_idx)), base_src, base_time,
                                max_per_traj=args.tube_max_per_traj, min_gap=1)
    base_sel = param_stratify(base_sel, traj_cell[base_src], tube_n, grid * grid, rng, priority=None)
    bi, bt = base_src[base_sel], base_time[base_sel]
    base = frames[bi, bt].astype(np.float32)                  # (M,L)
    base_par = params[bi].astype(np.float32)
    base_rms = np.sqrt(np.mean(base.astype(np.float64) ** 2, axis=1, keepdims=True)).astype(np.float32)
    print(f"tube base: {len(base)} attractor frames from {len(np.unique(bi))} trajectories "
          f"(t in [{tp1 // 2},{T}]); base RMS med={np.median(base_rms):.3f}")

    for band, (k1, k2) in TUBE_BANDS.items():
        for rho in args.rho:
            tag = f"{rho}".replace(".", "p")
            name = f"val_tube_{band}_rho{tag}"
            delta = band_perturbation(len(base), L, k1, k2, rng) * (rho * base_rms)
            pert = (base + delta).astype(np.float32)
            nxt = solve_one_step(pde, pert, base_par, args.solver_batch, name)
            finite = np.isfinite(nxt).all(axis=1) & np.isfinite(pert).all(axis=1)
            n_drop = int((~finite).sum())
            pert_f, nxt_f, par_f, src_f = pert[finite], nxt[finite], base_par[finite], bi[finite]
            save_set(out / f"{name}.npz", pert_f[:, None, :], nxt_f[:, None, :], par_f)
            entry = set_stats(pert_f[:, None, :], nxt_f[:, None, :], par_f, src_f, edges0, edges1, grid)
            entry.update({"criterion": f"attractor + band-limited perturbation k in [{k1},{k2}], rho={rho}",
                          "band": band, "k_range": [k1, k2], "rho": rho, "n_nonfinite_dropped": n_drop})
            manifest["sets"][name] = entry
            if n_drop:
                print(f"  {name}: dropped {n_drop} nonfinite states")

    manifest["runtime_seconds"] = round(time.time() - t_start, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out / 'manifest.json'}  (total {manifest['runtime_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
