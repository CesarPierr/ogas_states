#!/usr/bin/env python
"""Build HARD / COVERAGE validation sets from an existing uniform validation bank.

Implements docs/theory.md §8.2 (rare-state tail) and §8.3 (loss-uniform / coverage).
The difficulty coordinate is *model-independent* (total variation of the state), so the
held-out sets are a fair, fixed yardstick that does not depend on any surrogate.

Outputs npz files with the same schema as the uniform validation set
(states0, params, trajectories) so they load via eval.load_validation_data. Each transition
is stored as a length-2 trajectory (state, next_state) -> one-step coverage metric.

Usage:
    python scripts/build_hard_validation.py <uniform.npz> <out_dir> [--top 0.05 0.01] [--ncov 4000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def total_variation(states: np.ndarray) -> np.ndarray:
    """Model-independent difficulty: periodic total variation per state. states: (N,1,L)."""
    s = states.reshape(states.shape[0], -1).astype(np.float64)
    tv = np.abs(np.diff(s, axis=-1)).sum(-1)
    tv += np.abs(s[:, 0] - s[:, -1])  # periodic wrap
    return tv


def flatten_transitions(traj: np.ndarray, params: np.ndarray):
    """traj (n, T+1, 1, L), params (n, P) -> state, next_state, param per transition."""
    n, tp1 = traj.shape[:2]
    steps = tp1 - 1
    state = traj[:, :-1].reshape(n * steps, *traj.shape[2:])
    nxt = traj[:, 1:].reshape(n * steps, *traj.shape[2:])
    par = np.repeat(params, steps, axis=0)
    return state.astype(np.float32), nxt.astype(np.float32), par.astype(np.float32)


def save_set(path: Path, state: np.ndarray, nxt: np.ndarray, par: np.ndarray) -> None:
    # length-2 trajectories: [state, next_state]
    traj = np.stack([state, nxt], axis=1).astype(np.float32)  # (N,2,1,L)
    np.savez_compressed(path, states0=state.astype(np.float32), params=par.astype(np.float32), trajectories=traj)
    print(f"  wrote {path}  ({len(state)} transitions)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("uniform_npz")
    ap.add_argument("out_dir")
    ap.add_argument("--top", nargs="+", type=float, default=[0.05, 0.01])
    ap.add_argument("--ncov", type=int, default=4000, help="size of the loss-uniform coverage set")
    ap.add_argument("--cov_bins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with np.load(args.uniform_npz) as d:
        traj, params = d["trajectories"], d["params"]
    state, nxt, par = flatten_transitions(traj, params)
    tv = total_variation(state)
    order = np.argsort(tv)  # ascending
    N = len(tv)
    print(f"bank: {N} transitions; TV min={tv.min():.1f} med={np.median(tv):.1f} max={tv.max():.1f}")

    # --- §8.2a rare-state tail by TV (RMSE-absolute hardness: high-amplitude/sharp) ---
    for q in args.top:
        k = max(1, int(round(q * N)))
        idx = order[-k:]
        tag = f"top{int(round(q*100))}" if q >= 0.01 else f"top{q}"
        save_set(out / f"val_hard_tv_{tag}.npz", state[idx], nxt[idx], par[idx])
        print(f"  val_hard_tv_{tag}: TV>={tv[idx].min():.1f} (vs median {np.median(tv):.1f})")

    # --- §8.2b rare-state tail by LOW target amplitude (NRMSE-relative hardness:
    #     near-fixed-point / low-energy states blow up the *relative* error) ---
    amp = np.sqrt(np.mean(nxt.reshape(N, -1).astype(np.float64) ** 2, axis=-1))
    amp_order = np.argsort(amp)  # ascending; hardest-for-NRMSE = smallest amplitude
    for q in args.top:
        k = max(1, int(round(q * N)))
        idx = amp_order[:k]
        tag = f"top{int(round(q*100))}" if q >= 0.01 else f"top{q}"
        save_set(out / f"val_hard_lowamp_{tag}.npz", state[idx], nxt[idx], par[idx])
        print(f"  val_hard_lowamp_{tag}: target_amp<={amp[idx].max():.3f} (vs median {np.median(amp):.3f})")

    # --- §8.3 loss-uniform / coverage: equal mass per TV-quantile bin ---
    edges = np.quantile(tv, np.linspace(0, 1, args.cov_bins + 1))
    bins = np.clip(np.digitize(tv, edges[1:-1]), 0, args.cov_bins - 1)
    per = max(1, args.ncov // args.cov_bins)
    pick = []
    for b in range(args.cov_bins):
        members = np.where(bins == b)[0]
        if len(members) == 0:
            continue
        pick.append(rng.choice(members, size=min(per, len(members)), replace=False))
    cov_idx = np.concatenate(pick)
    save_set(out / "val_cov.npz", state[cov_idx], nxt[cov_idx], par[cov_idx])
    print(f"  val_cov: {len(cov_idx)} transitions, uniform over {args.cov_bins} TV bins")


if __name__ == "__main__":
    main()
