from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PDEConfig:
    name: str = "burgers"
    backend: str = "al4pde"
    resolution: int = 256
    dt: float = 2e-4
    domain_length: float = 1.0
    viscosity_range: tuple[float, float] = (0.003, 0.03)
    param_ranges: list[list[float]] | None = None
    param_log_scale: list[bool] | None = None
    ic: dict[str, Any] = field(default_factory=lambda: {"modes": 5, "amplitude": 1.0})
    n_substeps: int = 1   # internal solver steps per training step (dt_inner = dt/n_substeps)
    n_warmup: int = 0     # outer steps to discard before recording IC-generated trajectories


@dataclass
class PoolConfig:
    rounds: int = 8
    trajectory_steps: int = 75
    n_trajectories: int = 64
    uniform_fraction: float = 0.25
    # How the non-uniform half of the pool is filled:
    #   "generator"   -> difficulty-conditioned generative model (the method)
    #   "random_tube" -> attractor anchors + random low-wavenumber perturbations
    #                    (no learning; the decisive generator-necessity baseline)
    #   "mined_ic"    -> draw candidate_factor x random ICs (free), keep the ones with
    #                    highest ensemble disagreement, solver-label only those
    #                    (selection-without-generation baseline at matched solver budget)
    strategy: str = "generator"
    tube_rho_min: float = 0.05   # random_tube: relative perturbation radius range
    tube_rho_max: float = 0.5
    tube_kmax: int = 12          # random_tube: highest perturbed wavenumber


@dataclass
class SurrogateConfig:
    model: str = "al4pde_unet1d"
    hidden: int = 64
    depth: int = 3
    ensemble_size: int = 1
    epochs_per_round: int = 20
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-5
    difference_weight: float = 0.3
    # Ensemble decorrelation: each member gets its own shuffled batch order, and
    # optionally a bootstrap resample of the pool, so disagreement reflects more
    # than initialization noise.
    ensemble_bootstrap: bool = True
    # Gaussian input-noise injection during training, in units of the state std
    # (0 = off). With uniform_fraction=1.0 + ddpm.enabled=false this is the
    # noise-injection baseline (Sanchez-Gonzalez et al. style).
    input_noise_std: float = 0.0
    # Loss-weighted replay: oversample hard transitions (by pretrain loss) during
    # surrogate training. Hard-mining baseline at zero extra solver cost.
    loss_weighted_replay: bool = False


@dataclass
class DDPMConfig:
    enabled: bool = True
    generator: str = "ddpm"
    mode: str = "conditional_loss"
    param_mode: str = "none"
    steps: int = 64
    train_epochs: int = 10
    batch_size: int = 128
    lr: float = 3e-4
    hidden: int = 64
    residual_blocks: int = 0
    kernel_size: int = 5
    loss_metric: str = "mse"
    # Difficulty signal used to bin the pool for quantile conditioning:
    #   "loss"         -> surrogate one-step error (loss_metric: mse/rmse/nrmse)
    #   "ensemble_var" -> epistemic disagreement (variance across ensemble members);
    #                     isolates the *reducible* uncertainty (needs ensemble_size > 1).
    difficulty_signal: str = "loss"
    loss_condition_scale: float = 1.0
    loss_proposal: str = "empirical"
    loss_quantile_min: float = 0.75
    loss_quantile_max: float = 1.5
    param_prior: str = "uniform"
    param_loss_weight: float = 1.0
    sample_temperature: float = 1.0
    sample_batch_size: int = 512
    generated_pool_mode: str = "transition"
    solver_batch_size: int = 256
    sample_weight_floor: float = 0.05
    sample_weight_power: float = 1.0
    # Quantile-bin conditioning (mode == "conditional_quantile"). The generator is
    # conditioned on a discrete difficulty bin instead of a continuous loss value.
    n_quantiles: int = 0
    quant_embed_dim: int = 0
    # Quantile sampling law used at generation time: one of
    # top1, top2, top3, top5, top_half, exp_bias, uniform.
    sample_strategy: str = "exp_bias"
    sample_strategy_temp: float = 0.5
    # --- conditioning strength ---------------------------------------------------------
    # Classifier-free guidance on the quantile label (mode == "conditional_quantile"):
    #   cfg_dropout -> per-sample probability of replacing the label embedding with a
    #                  learned NULL embedding during generator training (0 = off).
    #   cfg_scale   -> sampling-time guidance weight: out = null + cfg_scale*(cond-null);
    #                  1.0 = exactly the conditional prediction (legacy, one forward
    #                  pass per step); != 1.0 costs a second forward pass per step.
    # The NULL embedding parameter is only created when either knob is active, so
    # default configs produce byte-identical state_dicts; checkpoint loading tolerates
    # the missing/extra key (checkpoint.py).
    cfg_dropout: float = 0.0
    cfg_scale: float = 1.0
    # How the conditioning vector enters the denoiser:
    #   "concat" -> broadcast + concatenated as input channels (legacy)
    #   "film"   -> keeps the input concat AND adds per-residual-block FiLM
    #               scale-shift h*(1+gamma)+beta from a zero-initialised per-block
    #               Linear on the cond vector (requires residual_blocks > 0).
    cond_mode: str = "concat"
    # --- conditioning-signal hygiene -------------------------------------------------
    # Which losses label difficulty for generator binning:
    #   "pretrain"  -> previous round's surrogate on this (unseen) pool: honest,
    #                  out-of-sample difficulty
    #   "posttrain" -> just-trained surrogate on its own training pool (legacy;
    #                  deflated by overfit)
    difficulty_from: str = "pretrain"
    # Normalization statistics for the difficulty signal across rounds:
    #   "per_round" -> recompute q05/q95 each round (legacy; bin k drifts in meaning)
    #   "ema"       -> exponential moving average of q05/q95 (smooth, stationary-ish)
    #   "frozen"    -> freeze stats at the first generator round
    difficulty_stats: str = "ema"
    difficulty_stats_momentum: float = 0.7
    # --- anti-self-feeding -----------------------------------------------------------
    # Relative weight of generator-sourced pool states (source!=0) in the generator's
    # OWN training set. 1.0 = legacy (generator half-trains on its own previous
    # outputs); 0.3 anchors training to solver/uniform states; 0.0 = never train on
    # own outputs.
    self_train_weight: float = 0.3
    # --- distribution prior ----------------------------------------------------------
    # Base distribution of the flow/diffusion sampler:
    #   "white"    -> N(0, I) (legacy)
    #   "spectrum" -> coloured Gaussian matched to the mean power spectrum of the
    #                 UNIFORM-sourced pool states (physical prior; used in both
    #                 training and sampling so there is no train/sample mismatch)
    noise_prior: str = "spectrum"
    # Sampling mode:
    #   "scratch" -> integrate from pure base noise (legacy)
    #   "edit"    -> SDEdit-style: start from a real attractor anchor partially
    #                re-noised to time edit_t0 and integrate to 1 with difficulty
    #                conditioning. Generates difficulty-targeted states NEAR the
    #                data manifold; deviation radius is controlled by edit_t0.
    sample_mode: str = "scratch"
    edit_t0: float = 0.6
    # --- oversample-then-reject ------------------------------------------------------
    # Generate candidate_factor x more candidate states than needed, score them with
    # the surrogate ensemble's disagreement (forward passes only, zero solver cost),
    # keep the top 1/candidate_factor, and only then pay the solver step. Selection
    # by realized disagreement replaces (weak) conditioning as the steering mechanism.
    candidate_factor: int = 1
    # Realism gate applied before disagreement ranking: drop candidates whose total
    # variation exceeds gate x the p99 TV of uniform-sourced pool states (0 = off).
    realism_tv_gate: float = 0.0


@dataclass
class ValidationConfig:
    n_trajectories: int = 128
    trajectory_steps: int = 99
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    rollout_steps: int = 50
    path: str | None = None
    hard_dir: str | None = None  # dir of post-hoc hard/tube/cov sets; if set, logged per round under hard_val/
    epoch_n_trajectories: int = 128


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "poolbased-surrogate-1d"
    group: str = "default"


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs/default"
    pde: PDEConfig = field(default_factory=PDEConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _merge_dataclass(cls, data: dict[str, Any]):
    base = cls()
    for key, value in data.items():
        if not hasattr(base, key):
            raise ValueError(f"Unknown config key {cls.__name__}.{key}")
        setattr(base, key, value)
    return base


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = ExperimentConfig()
    for key, value in raw.items():
        if key == "pde":
            cfg.pde = _merge_dataclass(PDEConfig, value or {})
            cfg.pde.viscosity_range = tuple(cfg.pde.viscosity_range)
        elif key == "pool":
            cfg.pool = _merge_dataclass(PoolConfig, value or {})
        elif key == "surrogate":
            cfg.surrogate = _merge_dataclass(SurrogateConfig, value or {})
        elif key == "ddpm":
            cfg.ddpm = _merge_dataclass(DDPMConfig, value or {})
        elif key == "validation":
            cfg.validation = _merge_dataclass(ValidationConfig, value or {})
        elif key == "wandb":
            cfg.wandb = _merge_dataclass(WandbConfig, value or {})
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown config key {key}")
    return cfg
