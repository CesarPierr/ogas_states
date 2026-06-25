"""Generator construction, param scaling, and sampling for the lab."""
from __future__ import annotations

import inspect
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import WeightedRandomSampler
from typing import Any
from ..models.ddpm import DDPM1D, FlowMatching1D
from .common import choose_device


def _scale_params(params: np.ndarray, param_min: np.ndarray, param_max: np.ndarray) -> np.ndarray:
    return (2.0 * (params - param_min) / (param_max - param_min + 1e-8) - 1.0).astype(np.float32)


def _unscale_params(params: np.ndarray, param_min: np.ndarray, param_max: np.ndarray) -> np.ndarray:
    return (0.5 * (params + 1.0) * (param_max - param_min) + param_min).astype(np.float32)


def _generator_class(generator: str):
    return FlowMatching1D if generator.lower() in ("flow", "flow_matching", "fm") else DDPM1D


def accepted_model_kwargs(generator: str, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Subset of model_kwargs the generator constructor actually accepts.

    Used so sweep arms / CLIs can carry knobs for newer model branches (e.g.
    cfg_dropout, cfg_scale, cond_mode) without breaking older constructors, and
    so checkpoints only persist kwargs that shaped the trained architecture.
    """
    sig_params = inspect.signature(_generator_class(generator).__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params.values()):
        return dict(model_kwargs)
    return {k: v for k, v in model_kwargs.items() if k in sig_params}


def make_generator(
    generator: str,
    resolution: int,
    hidden: int,
    steps: int,
    loss_conditional: bool,
    param_dim: int,
    param_mode: str,
    residual_blocks: int,
    kernel_size: int,
    n_quantiles: int = 0,
    quant_embed_dim: int = 0,
    **model_kwargs: Any,
):
    """Build a generator; extra **model_kwargs are forwarded to the constructor
    only when it accepts them (see accepted_model_kwargs)."""
    cls = _generator_class(generator)
    accepted = accepted_model_kwargs(generator, model_kwargs)
    dropped = sorted(k for k in model_kwargs if k not in accepted)
    if dropped:
        print(f"  [make_generator] {cls.__name__} does not accept {dropped}; ignoring.", flush=True)
    model_kwargs = accepted
    return cls(
        resolution=resolution,
        hidden=hidden,
        steps=steps,
        loss_conditional=loss_conditional,
        param_dim=param_dim,
        param_mode=param_mode,
        residual_blocks=residual_blocks,
        kernel_size=kernel_size,
        n_quantiles=n_quantiles,
        quant_embed_dim=quant_embed_dim,
        **model_kwargs,
    )


def _balanced_sampler_from_quantiles(quantile_labels: np.ndarray, n_quantiles: int) -> WeightedRandomSampler:
    """Sample each quantile bin with equal expected frequency per epoch."""
    bin_counts = np.bincount(quantile_labels.astype(np.int64), minlength=n_quantiles).astype(np.float64)
    bin_counts = np.maximum(bin_counts, 1)
    weights = (1.0 / bin_counts[quantile_labels.astype(np.int64)]).astype(np.float32)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _sampling_strategies(n_quantiles: int) -> dict[str, Any]:
    """Return a dict of strategy_name → callable(n, rng) → int array of quantile labels."""
    q = n_quantiles
    return {
        "top1":      lambda n, rng: np.full(n, q - 1, dtype=np.int64),
        "top2":      lambda n, rng: rng.choice([q - 2, q - 1], size=n),
        "top3":      lambda n, rng: rng.choice(np.arange(max(0, q - 3), q), size=n),
        "top5":      lambda n, rng: rng.choice(np.arange(max(0, q - 5), q), size=n),
        "top_half":  lambda n, rng: rng.integers(q // 2, q, size=n),
        "uniform":   lambda n, rng: rng.integers(0, q, size=n),
        "exp_bias":  lambda n, rng: rng.choice(q, size=n, p=_softmax(np.arange(q, dtype=float) * 0.5)),
    }


@torch.no_grad()
def sample_loss_generator(
    checkpoint_path: str | Path,
    n_samples: int,
    quantile_labels: np.ndarray,
    params: np.ndarray,
    device_name: str = "auto",
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample states from the generator conditioned on quantile labels.

    Returns (states_physical, params_physical).
    states are returned in physical units (denormalised from z-norm training space).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = choose_device(device_name)
    param_min = np.asarray(checkpoint["param_min"], dtype=np.float32)
    param_max = np.asarray(checkpoint["param_max"], dtype=np.float32)
    state_mean = float(checkpoint["state_mean"])
    state_std = float(checkpoint["state_std"])
    n_quantiles = int(checkpoint.get("n_quantiles", 0))
    quant_embed_dim = int(checkpoint.get("quant_embed_dim", 0))
    resolution = int(json.loads(checkpoint["resolved_config"])["pde"]["resolution"])

    model = make_generator(
        generator=str(checkpoint["generator"]),
        resolution=resolution,
        hidden=int(checkpoint["hidden"]),
        steps=int(checkpoint["steps"]),
        loss_conditional=True,
        param_dim=len(param_min),
        param_mode=str(checkpoint["param_mode"]),
        residual_blocks=int(checkpoint["residual_blocks"]),
        kernel_size=int(checkpoint["kernel_size"]),
        n_quantiles=n_quantiles,
        quant_embed_dim=quant_embed_dim,
        **dict(checkpoint.get("model_kwargs") or {}),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    quantile_labels = np.asarray(quantile_labels, dtype=np.int64).reshape(-1)
    if len(quantile_labels) != n_samples:
        quantile_labels = np.resize(quantile_labels, n_samples)
    params = np.asarray(params, dtype=np.float32)
    if len(params) != n_samples:
        params = np.resize(params, (n_samples, params.shape[1]))
    params_scaled = _scale_params(params, param_min, param_max)

    ql_t = torch.from_numpy(quantile_labels).to(device)
    pt_t = torch.from_numpy(params_scaled).to(device) if model.use_param_cond else None
    gen_norm, gen_params_scaled = model.sample(
        n_samples, params=pt_t, device=device, temperature=temperature, quantile_label=ql_t
    )
    # Denormalise: states were z-normed during training, reverse that here.
    states = (gen_norm.cpu().numpy() * state_std + state_mean).astype(np.float32)
    if gen_params_scaled is not None:
        params = _unscale_params(gen_params_scaled.cpu().numpy(), param_min, param_max)
    return states, params.astype(np.float32)
