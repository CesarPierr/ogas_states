# AGENT.md

## Scope

This repo is a small autonomous experiment for 1D pool-based surrogate learning.
Keep it simple. Do not import Melissa, ICML, or `apebench_online`.

Root:

```text
/home/cesarpi-ext/poolbased_surrogate_1d
```

## Purpose

Compare:

1. uniform trajectory sampling at every round
2. mixed sampling: some uniform trajectories plus DDPM-generated states

One sample is one transition:

```text
(state_t, pde_parameter) -> state_{t+1}
```

Required invariant:

```text
n_samples = pool.trajectory_steps * pool.n_trajectories
```

Round 0 always uses only uniform trajectories. Later rounds may replace part of
the pool with generated states advanced by one PDE step.

## Commands

Install env:

```bash
cd /home/cesarpi-ext/poolbased_surrogate_1d
scripts/install_uv_env.sh
source /hoyt/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv/bin/activate
```

Smoke:

```bash
python -m poolbased_surrogate.run configs/smoke.yaml
python -m poolbased_surrogate.run configs/smoke_weighted.yaml
```

Resume:

```bash
python -m poolbased_surrogate.run configs/default_1d.yaml --resume
```

Kraken devel:

```bash
KRAKEN_WALLTIME=00:30:00 KRAKEN_NUM_GPUS=1 scripts/submit_kraken_devel.sh configs/devel_uniform.yaml
KRAKEN_WALLTIME=00:30:00 KRAKEN_NUM_GPUS=1 scripts/submit_kraken_devel.sh configs/devel_mixed.yaml
```

If devel quota full:

```bash
KRAKEN_OAR_TYPE= KRAKEN_WALLTIME=00:30:00 KRAKEN_NUM_GPUS=1 scripts/submit_kraken_devel.sh configs/devel_uniform.yaml
```

## Main Files

```text
poolbased_surrogate/config.py            dataclass YAML config
poolbased_surrogate/pde.py               1D periodic PDE solvers and samplers
poolbased_surrogate/data.py              transition pool and datasets
poolbased_surrogate/models/surrogate.py  Conv1D surrogate and ensemble wrapper
poolbased_surrogate/models/ddpm.py       Conv1D DDPM
poolbased_surrogate/train.py             train surrogate/DDPM, propose losses
poolbased_surrogate/eval.py              Halton validation and rollout metrics
poolbased_surrogate/run.py               full experiment loop and checkpointing
```

## DDPM Modes

`conditional_loss`:

```text
p(state | loss)
```

Loss proposal samples from previous normalized losses plus small noise.

`weighted_unconditional`:

```text
p(state)
```

Training weights each state by `0.05 + normalized_loss`, giving density biased
toward high-loss states.

## Checkpointing

Checkpoint:

```text
<output_dir>/checkpoint_latest.pt
```

Saves after each completed round:

- next round index
- surrogate weights
- DDPM weights
- transition pool
- losses
- history
- NumPy and Torch RNG states
- WandB run id

Resume granularity is round-level. Mid-round walltime loss repeats the current
round from the last completed checkpoint.

## WandB

WandB is optional in config. Resume reuses same run id.

Useful metrics:

- `val/nrmse_mean`
- `rollout/nrmse_final`
- `pool/loss_mean`
- `pool/loss_p90`
- `train/loss`
- `ddpm/loss`

## Comparison Configs

Uniform baseline:

```text
configs/compare_uniform.yaml
```

Mixed DDPM conditional:

```text
configs/compare_mixed_conditional.yaml
```

Both use same seed and validation setup. Only sampling strategy differs.

## Rules For Future Edits

- Keep code standalone.
- Keep config surface small.
- Preserve `n_samples = T * n_trajectories`.
- Prefer simple NumPy/Torch over framework abstractions.
- Add tests through smoke configs, not heavy infrastructure.
- Do not commit `runs/`, checkpoints, `.egg-info`, or caches.
