from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import TransitionPool
from .eval import evaluate
from .models.ddpm import DDPM1D
from .models.surrogate import build_surrogate
from .pde import PDE1D
from .train import propose_losses, train_ddpm, train_surrogate


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_uniform_pool(pde: PDE1D, n_traj: int, steps: int, rng: np.random.Generator) -> TransitionPool:
    params = pde.sample_params_uniform(n_traj, rng)
    states0 = pde.sample_ic_uniform(n_traj, rng)
    traj = pde.simulate(states0, params, steps)
    return TransitionPool.from_trajectories(traj, params)


@torch.no_grad()
def make_mixed_pool(
    pde: PDE1D,
    ddpm: DDPM1D,
    previous: TransitionPool,
    n_traj: int,
    steps: int,
    uniform_fraction: float,
    ddpm_mode: str,
    rng: np.random.Generator,
    device: torch.device,
) -> TransitionPool:
    n_uniform = int(round(n_traj * uniform_fraction))
    n_generated_traj = n_traj - n_uniform
    pools = []
    if n_uniform > 0:
        pools.append(make_uniform_pool(pde, n_uniform, steps, rng))
    if n_generated_traj > 0:
        n_generated = n_generated_traj * steps
        params = pde.sample_params_uniform(n_generated, rng)
        if previous.losses is None:
            loss_values = rng.uniform(0.5, 1.0, size=(n_generated, 1)).astype(np.float32)
        else:
            loss_values = propose_losses(previous.losses, n_generated, rng)
        cond = torch.from_numpy(loss_values).to(device) if ddpm_mode == "conditional_loss" else None
        states0 = ddpm.sample(n_generated, cond, device=device).cpu().numpy().astype(np.float32)
        next_state = pde.step(states0[:, 0].astype(np.float64), params[:, 0])[:, None, :].astype(np.float32)
        pools.append(TransitionPool(states0, params, next_state))
    states = np.concatenate([p.states for p in pools], axis=0)
    params = np.concatenate([p.params for p in pools], axis=0)
    next_states = np.concatenate([p.next_states for p in pools], axis=0)
    return TransitionPool(states, params, next_states)


def maybe_wandb(cfg):
    if not cfg.wandb.enabled:
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        config=asdict(cfg),
    )


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit("Usage: python -m poolbased_surrogate.run <config.yaml>")
    cfg = load_config(argv[0])
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = choose_device(cfg.device)

    pde = PDE1D(cfg.pde)
    model = build_surrogate(
        resolution=cfg.pde.resolution,
        hidden=cfg.surrogate.hidden,
        depth=cfg.surrogate.depth,
        ensemble_size=cfg.surrogate.ensemble_size,
    ).to(device)
    ddpm = DDPM1D(
        resolution=cfg.pde.resolution,
        hidden=cfg.ddpm.hidden,
        steps=cfg.ddpm.steps,
        conditional=cfg.ddpm.mode == "conditional_loss",
    ).to(device)
    run = maybe_wandb(cfg)

    pool = make_uniform_pool(pde, cfg.pool.n_trajectories, cfg.pool.trajectory_steps, rng)
    history = []
    for round_id in range(cfg.pool.rounds):
        if round_id > 0:
            pool = make_mixed_pool(
                pde=pde,
                ddpm=ddpm,
                previous=pool,
                n_traj=cfg.pool.n_trajectories,
                steps=cfg.pool.trajectory_steps,
                uniform_fraction=cfg.pool.uniform_fraction,
                ddpm_mode=cfg.ddpm.mode,
                rng=rng,
                device=device,
            )
        train_metrics = train_surrogate(
            model=model,
            pool=pool,
            epochs=cfg.surrogate.epochs_per_round,
            batch_size=cfg.surrogate.batch_size,
            lr=cfg.surrogate.lr,
            weight_decay=cfg.surrogate.weight_decay,
            device=device,
        )
        ddpm_metrics = {}
        if cfg.ddpm.enabled:
            ddpm_metrics = train_ddpm(
                ddpm=ddpm,
                pool=pool,
                epochs=cfg.ddpm.train_epochs,
                batch_size=cfg.ddpm.batch_size,
                lr=cfg.ddpm.lr,
                device=device,
                mode=cfg.ddpm.mode,
            )
        eval_metrics = evaluate(
            model=model,
            pde=pde,
            n_trajectories=cfg.validation.n_trajectories,
            trajectory_steps=cfg.validation.trajectory_steps,
            rollout_steps=cfg.validation.rollout_steps,
            quantiles=cfg.validation.quantiles,
            seed=cfg.seed + 10_000,
            device=device,
        )
        metrics = {
            "round": round_id,
            "pool/n_samples": len(pool),
            "pool/loss_mean": float(np.mean(pool.losses)) if pool.losses is not None else 0.0,
            "pool/loss_p90": float(np.quantile(pool.losses, 0.9)) if pool.losses is not None else 0.0,
            **train_metrics,
            **ddpm_metrics,
            **eval_metrics,
        }
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))
        if run is not None:
            run.log(metrics, step=round_id)
        np.savez_compressed(out / f"pool_round_{round_id}.npz", states=pool.states, params=pool.params, next_states=pool.next_states, losses=pool.losses)
        torch.save(model.state_dict(), out / "surrogate.pt")
        torch.save(ddpm.state_dict(), out / "ddpm.pt")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
