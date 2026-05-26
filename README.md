# Pool-Based Surrogate 1D

Minimal autonomous repo for pool-based 1D PDE surrogate experiments.

No Melissa, no ICML runtime dependency. The code keeps only simple ideas:

- sample PDE parameters and initial conditions
- simulate 1D trajectories
- train one-step surrogate on transition pool
- store per-transition recent losses
- train conditional DDPM for `p(state | loss)`
- or train unconditional DDPM with state density weighted proportional to loss
- next rounds mix uniform new trajectories and DDPM-generated states
- evaluate after each round on Halton validation trajectories
- optional WandB logging

## Quick Start

```bash
cd /home/cesarpi-ext/poolbased_surrogate_1d
/hoyt/PROJECTS/pr-melissa/cesarpi-ext/.melissa-apebench-xp2-venv/bin/python -m poolbased_surrogate.run configs/smoke.yaml
```

Weighted unconditional DDPM smoke:

```bash
/hoyt/PROJECTS/pr-melissa/cesarpi-ext/.melissa-apebench-xp2-venv/bin/python -m poolbased_surrogate.run configs/smoke_weighted.yaml
```

For real runs, edit `configs/default_1d.yaml`.

## Core Assumption

`n_samples = trajectory_steps * n_trajectories`.

Round 0 uses only uniform initial conditions and PDE parameters. Later rounds use:

- `uniform_fraction` full uniform trajectories
- `1 - uniform_fraction` DDPM-generated states conditioned on proposed loss values
- for `ddpm.mode=weighted_unconditional`, generated states come from `p(state)` after
  loss-weighted training instead of explicit loss conditioning

Each generated state is advanced by the PDE for one transition, then replaces the previous pool.

## Config Knobs

- `pde.name`: `burgers`, `advection`, `diffusion`, `ks`
- `pool.rounds`: number of pool replacement rounds
- `pool.trajectory_steps * pool.n_trajectories`: transition count per round
- `pool.uniform_fraction`: fraction kept as full uniform trajectories after round 0
- `surrogate.ensemble_size`: `>1` enables uncertainty computation hooks
- `ddpm.mode`: `conditional_loss` or `weighted_unconditional`
- `validation.*`: Halton validation size, quantiles, rollout horizon

## Files

- `poolbased_surrogate/pde.py`: from-scratch 1D periodic spectral PDEs
- `poolbased_surrogate/models/surrogate.py`: small ConvUNet-style 1D surrogate
- `poolbased_surrogate/models/ddpm.py`: convolutional 1D DDPM
- `poolbased_surrogate/run.py`: full round loop
