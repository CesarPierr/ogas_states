#!/usr/bin/env python
"""Concatenate multiple validation datasets (.npz) into one massive bank."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .npz files")
    parser.add_argument("--output", required=True, help="Output .npz file")
    args = parser.parse_args()

    all_states0 = []
    all_params = []
    all_trajs = []

    for f in sorted(args.inputs):
        print(f"Loading {f}...")
        with np.load(f) as d:
            all_states0.append(d["states0"])
            all_params.append(d["params"])
            all_trajs.append(d["trajectories"])
    
    states0 = np.concatenate(all_states0, axis=0)
    params = np.concatenate(all_params, axis=0)
    trajectories = np.concatenate(all_trajs, axis=0)

    print(f"Concatenated shapes:")
    print(f"  states0:      {states0.shape}")
    print(f"  params:       {params.shape}")
    print(f"  trajectories: {trajectories.shape}")

    print(f"Saving to {args.output}...")
    np.savez_compressed(
        args.output,
        states0=states0,
        params=params,
        trajectories=trajectories
    )
    print("Done!")

if __name__ == "__main__":
    main()
