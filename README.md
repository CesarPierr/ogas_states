# Pool-Based Surrogate 1D

Autonomous experimental repo for 1D pool-based active surrogate learning.

Goal: test whether a generative state sampler can replace part of fresh uniform
trajectory generation while keeping or improving surrogate quality.

No Melissa. No ICML runtime dependency. Code is intentionally small and easy to
modify.

## Onboarding

If you are discovering the project, start with the guided mini-MOOC:

```text
docs/mooc_decouverte.md
notebooks/mooc_00_project_map.ipynb
notebooks/mooc_01_data_pde_pool.ipynb
notebooks/mooc_02_results_investigations.ipynb
notebooks/mooc_03_research_next_steps.ipynb
```

## Pipeline

One data sample is one transition:

```text
(state_t, pde_parameter) -> state_{t+1}
```

Each round owns exactly:

```text
n_samples = pool.trajectory_steps * pool.n_trajectories
```

Round 0:

1. sample `pool.n_trajectories` PDE parameters uniformly
2. sample `pool.n_trajectories` Fourier initial conditions uniformly
3. simulate full trajectories with the selected PDE
4. flatten trajectories into transitions
5. train the surrogate for `surrogate.epochs_per_round`
6. compute and store recent per-transition surrogate losses
7. train DDPM on states using the stored losses
8. evaluate on Halton validation trajectories
9. save checkpoint

Rounds `>= 1`:

1. create a new pool from scratch
2. generate `pool.uniform_fraction` of trajectories with uniform ICs and uniform parameters
3. generate the remaining states with DDPM
4. sample fresh PDE parameters for generated states
5. advance every generated state by one PDE step to get `state_{t+1}`
6. replace the full previous pool with this new pool
7. train surrogate, recompute losses, train DDPM, validate, checkpoint

## Sampling Modes

### Uniform Baseline

Config:

```yaml
pool:
  uniform_fraction: 1.0
ddpm:
  enabled: false
```

Every round is a fresh uniform trajectory pool.

### Conditional DDPM

Config:

```yaml
pool:
  uniform_fraction: 0.25
ddpm:
  enabled: true
  mode: conditional_loss
```

DDPM learns:

```text
p(state | loss)
```

Loss proposal currently samples from empirical normalized previous losses plus
small Gaussian noise. Generated states are then advanced one PDE step.

### Weighted Unconditional DDPM

Config:

```yaml
ddpm:
  mode: weighted_unconditional
```

DDPM learns:

```text
p(state)
```

but denoising loss is weighted by:

```text
0.05 + normalized_transition_loss
```

This creates a state density biased toward high-loss regions, similar in spirit
to DAS-PINN style density proportional to residual/loss.

## PDEs

Implemented from scratch in `poolbased_surrogate/pde.py`.

All PDEs are 1D periodic and use spectral derivatives through NumPy FFT.

Available names:

- `burgers`
- `advection`
- `diffusion`
- `ks` or `kuramoto_sivashinsky`

Main PDE config:

```yaml
pde:
  name: burgers
  resolution: 128
  dt: 0.0005
  viscosity_range: [0.005, 0.03]
  velocity_range: [-2.0, 2.0]
  diffusion_range: [0.002, 0.05]
  ic:
    modes: 4
    amplitude: 1.0
```

For `burgers`, the parameter is viscosity. For `advection`, velocity. For
`diffusion`, diffusivity. KS currently uses a dummy scalar parameter to preserve
the common interface.

## Initial Conditions

Initial conditions are random truncated Fourier series:

```text
u(x) = sum_m a_m sin(2 pi m x) + b_m cos(2 pi m x)
```

with coefficients decaying as `1 / m`, then normalized by max absolute value.

Halton validation ICs use the same basis but deterministic low-discrepancy
coefficients.

## Surrogate Architecture

File:

```text
poolbased_surrogate/models/surrogate.py
```

Current model: small ConvUNet-style residual 1D model.

Input:

```text
state:  [B, 1, X]
params: [B, 1]
```

Parameter conditioning is done by expanding the scalar parameter over the spatial
grid and concatenating it as an input channel:

```text
[state, parameter_channel] -> Conv1D blocks -> delta_state
prediction = state + delta_state
```

Config:

```yaml
surrogate:
  model: conv_unet
  hidden: 48
  depth: 3
  ensemble_size: 1
  epochs_per_round: 5
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0
```

`ensemble_size > 1` builds several independent models and averages predictions.
Uncertainty hooks exist through ensemble prediction variance, but the first
comparison uses loss-based DDPM conditioning.

## DDPM Architecture

File:

```text
poolbased_surrogate/models/ddpm.py
```

DDPM is fully convolutional in state dimension.

Input:

```text
noisy_state: [B, 1, X]
timestep:    [B]
loss cond:   [B, 1] optional
```

Conditioning:

- sinusoidal timestep embedding
- optional scalar loss appended to embedding
- MLP maps conditioning to `hidden` channels
- conditioning channels expanded over spatial dimension
- concatenated with noisy state
- Conv1D denoiser predicts noise

Config:

```yaml
ddpm:
  enabled: true
  mode: conditional_loss
  steps: 32
  train_epochs: 3
  batch_size: 64
  lr: 0.001
  hidden: 48
```

Modes:

- `conditional_loss`: condition on normalized loss
- `weighted_unconditional`: no explicit condition, high-loss states weighted more

## Validation

Validation happens after each round.

Validation data is generated on the fly using Halton sampling:

- Halton PDE parameters
- Halton Fourier IC coefficients
- true PDE rollouts

Metrics:

- one-step RMSE mean
- one-step NRMSE mean
- RMSE/NRMSE by difficulty quantiles
- rollout RMSE mean/final
- rollout NRMSE mean/final

Config:

```yaml
validation:
  n_trajectories: 32
  trajectory_steps: 24
  quantiles: [0.25, 0.5, 0.75, 0.9]
  rollout_steps: 24
```

Difficulty quantile currently uses target energy:

```text
mean(state_{t+1}^2)
```

This is simple and stable; replace it if another difficulty score is more useful.

## WandB

Config:

```yaml
wandb:
  enabled: true
  project: poolbased-surrogate-1d
  group: compare_mixed_conditional
```

Logged every round:

- `pool/n_samples`
- `pool/loss_mean`
- `pool/loss_p90`
- `train/loss`
- `ddpm/loss`
- all validation and rollout metrics

Resume keeps the same WandB run id, so walltime restarts continue the same run.

## Resume

Runs save:

```text
<output_dir>/checkpoint_latest.pt
```

after each completed round.

Resume:

```bash
python -m poolbased_surrogate.run configs/default_1d.yaml --resume
```

Restored:

- next round index
- surrogate weights
- DDPM weights
- transition pool
- per-transition losses
- history
- NumPy RNG state
- Torch RNG state
- CUDA RNG state when available
- WandB run id

Resume granularity is round-level. If a walltime kills the job mid-round, the
last completed round is reused.

## Install

Using uv:

```bash
cd /home/cesarpi-ext/ogas_states
scripts/install_uv_env.sh
source /bettik/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv/bin/activate
```

Default env path:

```text
/bettik/PROJECTS/pr-melissa/cesarpi-ext/.poolbased-surrogate-venv
```

Override:

```bash
POOLBASED_ENV=/path/to/env scripts/install_uv_env.sh
```

## Local Runs

Smoke conditional:

```bash
python -m poolbased_surrogate.run configs/smoke.yaml
```

Smoke weighted:

```bash
python -m poolbased_surrogate.run configs/smoke_weighted.yaml
```

Run both:

```bash
scripts/run_smoke.sh
```

Main comparison:

```bash
python -m poolbased_surrogate.run configs/compare_uniform.yaml --resume
python -m poolbased_surrogate.run configs/compare_mixed_conditional.yaml --resume
```

## Bigfoot GPU

Submit to OAR on Bigfoot. By default this pre-creates or verifies the shared
validation dataset before submitting, so compared experiments use the same
validation trajectories.

```bash
scripts/install_uv_env.sh
BIGFOOT_WALLTIME=08:00:00 BIGFOOT_NUM_GPUS=1 BIGFOOT_GPU_MODEL=A100 scripts/submit_bigfoot.sh configs/bigfoot_uniform.yaml
BIGFOOT_WALLTIME=08:00:00 BIGFOOT_NUM_GPUS=1 BIGFOOT_GPU_MODEL=A100 scripts/submit_bigfoot.sh configs/bigfoot_mixed.yaml
```

For the Bigfoot devel sandbox, request a MIG partition:

```bash
BIGFOOT_OAR_TYPE=devel BIGFOOT_WALLTIME=00:30:00 scripts/submit_bigfoot.sh configs/bigfoot_mixed.yaml
```

The wrapper runs:

```bash
python -m poolbased_surrogate.run <config> --resume
```

Logs:

```text
/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_oar/<jobid>.out
/bettik/PROJECTS/pr-melissa/cesarpi-ext/poolbased_surrogate_oar/<jobid>.err
```

## Main Comparison

Question:

```text
Is mixed uniform + generative state sampling better than uniform-only sampling?
```

Configs:

- `configs/compare_uniform.yaml`
- `configs/compare_mixed_conditional.yaml`

Both use same seed and validation. Difference:

```yaml
# uniform
pool.uniform_fraction: 1.0
ddpm.enabled: false

# mixed
pool.uniform_fraction: 0.25
ddpm.enabled: true
ddpm.mode: conditional_loss
```

Look at:

- `val/nrmse_mean`
- `rollout/nrmse_final`
- `pool/loss_p90`
- convergence per round

## Output Files

Each run directory contains:

```text
config.resolved.json
checkpoint_latest.pt
history.json
pool_round_<round>.npz
surrogate.pt
ddpm.pt
```

`pool_round_<round>.npz` stores:

- `states`
- `params`
- `next_states`
- `losses`

## Repo Map

```text
poolbased_surrogate/config.py          dataclass config loader
poolbased_surrogate/pde.py             1D PDEs and samplers
poolbased_surrogate/data.py            transition pool datasets
poolbased_surrogate/models/surrogate.py Conv1D surrogate and ensemble
poolbased_surrogate/models/ddpm.py      Conv1D DDPM
poolbased_surrogate/train.py           training helpers
poolbased_surrogate/eval.py            validation and rollout metrics
poolbased_surrogate/run.py             experiment entry point
scripts/install_uv_env.sh              uv env install
scripts/submit_bigfoot.sh              Bigfoot OAR GPU submit
scripts/submit_kraken_devel.sh         legacy Kraken OAR devel submit
```

## Design Choices

Code is experimental, not production:

- simple YAML config
- no registry framework
- no plugin system
- no distributed training
- no multiprocessing
- no external dataset format
- round-level checkpointing only

This should keep modifications cheap while making the active sampling idea easy
to inspect.
