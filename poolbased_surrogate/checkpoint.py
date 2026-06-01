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
        },
        "history": history,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "wandb_run_id": wandb_run_id,
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, ddpm, rng: np.random.Generator, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["surrogate"])
    ddpm.load_state_dict(payload["ddpm"])
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
    )
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "torch_cuda_rng_state" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state"])
    return int(payload["next_round"]), pool, list(payload["history"]), str(payload["wandb_run_id"])
