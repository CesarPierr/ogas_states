"""Round checkpoint save/load: surrogate + generator weights, pool, and RNG state."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import TransitionPool


def save_checkpoint(
    path: Path,
    round_id: int,
    model,
    ddpm,
    pool: TransitionPool,
    history: list[dict[str, float]],
    rng: np.random.Generator,
    wandb_run_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_round": round_id + 1,
        "surrogate": model.state_dict(),
        "ddpm": ddpm.state_dict(),
        "pool": {
            "states": torch.from_numpy(pool.states),
            "params": torch.from_numpy(pool.params),
            "next_states": torch.from_numpy(pool.next_states),
            "losses": None if pool.losses is None else torch.from_numpy(pool.losses),
            "source": None if pool.source is None else torch.from_numpy(pool.source),
            "proposed_losses": None
            if pool.proposed_losses is None
            else torch.from_numpy(pool.proposed_losses),
            "pretrain_losses": None
            if pool.pretrain_losses is None
            else torch.from_numpy(pool.pretrain_losses),
            "uncertainty": None if pool.uncertainty is None else torch.from_numpy(pool.uncertainty),
            "pretrain_uncertainty": None
            if pool.pretrain_uncertainty is None
            else torch.from_numpy(pool.pretrain_uncertainty),
            "target_bins": None if pool.target_bins is None else torch.from_numpy(pool.target_bins),
        },
        "history": history,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "wandb_run_id": wandb_run_id,
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(payload, tmp, _use_new_zipfile_serialization=False)
    except Exception:
        torch.save(payload, tmp)
    tmp.replace(path)


# Generator parameters that may legitimately differ between checkpoints and the
# constructed model: the CFG NULL label embedding (only created when cfg_dropout>0
# or cfg_scale!=1) and the per-block FiLM layers (only created when cond_mode=film).
_OPTIONAL_GENERATOR_KEYS = ("null_quant_embed", ".film.")


def load_generator_state_dict(ddpm, state_dict) -> None:
    """Tolerant generator load: optional conditioning params (CFG/FiLM) may be
    missing from old checkpoints or extra when loading new checkpoints into a
    default model. Any other key mismatch is still a hard error."""
    result = ddpm.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in result.missing_keys if not any(t in k for t in _OPTIONAL_GENERATOR_KEYS)]
    bad_unexpected = [k for k in result.unexpected_keys if not any(t in k for t in _OPTIONAL_GENERATOR_KEYS)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Generator state_dict mismatch: missing={bad_missing} unexpected={bad_unexpected}"
        )
    if result.missing_keys or result.unexpected_keys:
        print(
            "checkpoint: tolerated optional generator keys "
            f"(missing={list(result.missing_keys)}, unexpected={list(result.unexpected_keys)})"
        )


def load_checkpoint(path: Path, model, ddpm, rng: np.random.Generator, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["surrogate"])
    load_generator_state_dict(ddpm, payload["ddpm"])
    p = payload["pool"]
    pool = TransitionPool(
        states=p["states"].cpu().numpy(),
        params=p["params"].cpu().numpy(),
        next_states=p["next_states"].cpu().numpy(),
        losses=None if p["losses"] is None else p["losses"].cpu().numpy(),
        source=None if p.get("source") is None else p["source"].cpu().numpy(),
        proposed_losses=None
        if p.get("proposed_losses") is None
        else p["proposed_losses"].cpu().numpy(),
        pretrain_losses=None
        if p.get("pretrain_losses") is None
        else p["pretrain_losses"].cpu().numpy(),
        uncertainty=None if p.get("uncertainty") is None else p["uncertainty"].cpu().numpy(),
        pretrain_uncertainty=None
        if p.get("pretrain_uncertainty") is None
        else p["pretrain_uncertainty"].cpu().numpy(),
        target_bins=None if p.get("target_bins") is None else p["target_bins"].cpu().numpy(),
    )
    if hasattr(ddpm, "restore_generator_state"):
        ddpm.restore_generator_state()
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "torch_cuda_rng_state" in payload:
        # map_location=device may have moved the saved ByteTensors to CUDA;
        # set_rng_state_all requires CPU ByteTensors.
        torch.cuda.set_rng_state_all([s.cpu() for s in payload["torch_cuda_rng_state"]])
    return int(payload["next_round"]), pool, list(payload["history"]), str(payload["wandb_run_id"])
