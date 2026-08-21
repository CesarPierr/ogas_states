"""Entry point: parse config, build the models, and run the active-learning rounds.

Per-round work lives in dedicated modules: ``pool`` (build the mixed pool),
``train`` (surrogate + generator), ``generator_eval`` (generator realism),
``eval`` (validation/rollout), ``metrics`` (pool diagnostics), ``viz`` +
``wandb_logging`` (figures/logging), and ``checkpoint`` (resume state).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .eval import (
    create_validation_data,
    evaluate_one_step_validation,
    evaluate_validation,
    load_hard_validation_sets,
    load_validation_data,
    save_validation_data,
)
from .generator_eval import evaluate_generator
from .metrics import pool_debug_metrics
from .models.ddpm import DDPM1D, FlowMatching1D
from .models.surrogate import build_surrogate
from .pde import PDE1D
from .pool import make_mixed_pool, make_uniform_pool
from .train import (
    compute_transition_losses,
    compute_transition_uncertainty,
    train_ddpm,
    train_surrogate,
)
from .wandb_logging import log_wandb_diagnostics, make_epoch_callback, maybe_wandb


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def apply_override(cfg, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use key=value syntax, got {expression!r}")
    key, raw_value = expression.split("=", 1)
    target = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown override key {key!r}")
        target = getattr(target, part)
    final = parts[-1]
    if not hasattr(target, final):
        raise ValueError(f"Unknown override key {key!r}")
    value = yaml.safe_load(raw_value)
    current = getattr(target, final)
    if isinstance(current, tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(target, final, value)


def validation_path(cfg, out: Path) -> Path:
    if cfg.validation.path:
        return Path(cfg.validation.path)
    return out / "validation.npz"


def load_or_create_validation(cfg, pde: PDE1D, out: Path):
    path = validation_path(cfg, out)
    if path.exists():
        validation = load_validation_data(path)
        if (
            validation.n_trajectories == cfg.validation.n_trajectories
            and validation.trajectory_steps == cfg.validation.trajectory_steps
            and validation.states0.shape[-1] == cfg.pde.resolution
        ):
            print(f"loaded validation set from {path}")
            return validation
        raise ValueError(
            f"Validation set {path} does not match config: "
            f"file has {validation.n_trajectories} trajectories x "
            f"{validation.trajectory_steps} steps at resolution {validation.states0.shape[-1]}, "
            f"config asks for {cfg.validation.n_trajectories} trajectories x "
            f"{cfg.validation.trajectory_steps} steps at resolution {cfg.pde.resolution}."
        )
    validation = create_validation_data(
        pde=pde,
        n_trajectories=cfg.validation.n_trajectories,
        trajectory_steps=cfg.validation.trajectory_steps,
        seed=cfg.seed + 10_000,
    )
    save_validation_data(path, validation)
    print(
        "created validation set "
        f"{path} with {validation.n_trajectories} trajectories x {validation.trajectory_steps} steps"
    )
    return validation


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir/checkpoint_latest.pt")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing checkpoint")
    parser.add_argument(
        "--create-validation-only",
        action="store_true",
        help="Create or validate the configured validation dataset and exit",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set seed=101 --set pool.rounds=4",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for expression in args.overrides:
        apply_override(cfg, expression)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2))

    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    pde = PDE1D(cfg.pde)
    validation = load_or_create_validation(cfg, pde, out)
    if args.create_validation_only:
        return

    # Optional fixed hard/tube/coverage sets (post-hoc yardstick); logged each round under hard_val/.
    hard_sets = load_hard_validation_sets(cfg.validation.hard_dir) if cfg.validation.hard_dir else {}
    if hard_sets:
        print(f"loaded {len(hard_sets)} hard/tube validation sets: {sorted(hard_sets)}", flush=True)

    # Validation-set statistics used to z-normalize states for DDPM training/generation,
    # so generated states stay within the physical range of the data.
    state_mean = float(validation.trajectories.mean())
    state_std = float(validation.trajectories.std()) + 1e-8
    print(f"state z-norm from validation: mean={state_mean:.4f} std={state_std:.4f}")

    # Real-KS reference states (skip t=0) for the per-round generator realism metrics.
    _val_traj = validation.trajectories
    _val_states_flat = _val_traj[:, 1:].reshape(-1, 1, _val_traj.shape[-1]).astype(np.float32)
    _gen_eval_idx = rng.choice(len(_val_states_flat), size=min(1024, len(_val_states_flat)), replace=False)
    val_ref_states = _val_states_flat[_gen_eval_idx]

    device = choose_device(cfg.device)
    model = build_surrogate(
        resolution=cfg.pde.resolution,
        hidden=cfg.surrogate.hidden,
        depth=cfg.surrogate.depth,
        ensemble_size=cfg.surrogate.ensemble_size,
        param_dim=len(pde.param_ranges),
        model_name=cfg.surrogate.model,
        difference_weight=cfg.surrogate.difference_weight,
    ).to(device)
    generator_cls = FlowMatching1D if str(cfg.ddpm.generator).lower() in ("flow", "flow_matching", "fm") else DDPM1D
    ddpm = generator_cls(
        resolution=cfg.pde.resolution,
        hidden=cfg.ddpm.hidden,
        steps=cfg.ddpm.steps,
        loss_conditional=cfg.ddpm.mode == "conditional_loss",
        param_dim=len(pde.param_ranges),
        param_mode=cfg.ddpm.param_mode,
        residual_blocks=cfg.ddpm.residual_blocks,
        kernel_size=cfg.ddpm.kernel_size,
        n_quantiles=cfg.ddpm.n_quantiles if cfg.ddpm.mode == "conditional_quantile" else 0,
        quant_embed_dim=cfg.ddpm.quant_embed_dim,
        cfg_dropout=cfg.ddpm.cfg_dropout,
        cfg_scale=cfg.ddpm.cfg_scale,
        cond_mode=cfg.ddpm.cond_mode,
        amplitude_conditional=cfg.ddpm.amplitude_conditional,
    ).to(device)
    checkpoint = out / "checkpoint_latest.pt"
    start_round = 0
    pool = make_uniform_pool(pde, cfg.pool.n_trajectories, cfg.pool.trajectory_steps, rng)
    history: list[dict[str, float]] = []
    wandb_run_id = uuid.uuid4().hex
    if (
        not cfg.ddpm.enabled
        and cfg.pool.uniform_fraction < 1.0
        and str(cfg.pool.strategy) == "generator"
    ):
        raise ValueError(
            "ddpm.enabled=false with pool.strategy=generator requires pool.uniform_fraction=1.0 "
            "(use pool.strategy=random_tube or mined_ic for generator-free mixed pools)"
        )
    if args.resume and not args.fresh and checkpoint.exists():
        start_round, pool, history, wandb_run_id = load_checkpoint(checkpoint, model, ddpm, rng, device)
        print(f"resumed from {checkpoint} at round {start_round}")
    run = maybe_wandb(cfg, wandb_run_id)

    for round_id in range(start_round, cfg.pool.rounds):
        round_start_time = time.perf_counter()
        timing_metrics: dict[str, float] = {}

        def finish_phase(name: str, started_at: float) -> None:
            elapsed = time.perf_counter() - started_at
            timing_metrics[f"timing/{name}_sec"] = float(elapsed)
            print(f"round {round_id}: timing {name}={elapsed:.2f}s", flush=True)

        if round_id > 0:
            print(f"round {round_id}: generating mixed pool", flush=True)
            phase_start = time.perf_counter()
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
                state_mean=state_mean,
                state_std=state_std,
                loss_proposal=cfg.ddpm.loss_proposal,
                loss_metric=cfg.ddpm.loss_metric,
                loss_quantile_min=cfg.ddpm.loss_quantile_min,
                loss_quantile_max=cfg.ddpm.loss_quantile_max,
                param_prior=cfg.ddpm.param_prior,
                sample_temperature=cfg.ddpm.sample_temperature,
                loss_condition_scale=cfg.ddpm.loss_condition_scale,
                sample_batch_size=cfg.ddpm.sample_batch_size,
                generated_pool_mode=cfg.ddpm.generated_pool_mode,
                solver_batch_size=cfg.ddpm.solver_batch_size,
                sample_strategy=cfg.ddpm.sample_strategy,
                sample_strategy_temp=cfg.ddpm.sample_strategy_temp,
                surrogate=model,
                strategy=cfg.pool.strategy,
                tube_rho_min=cfg.pool.tube_rho_min,
                tube_rho_max=cfg.pool.tube_rho_max,
                tube_kmax=cfg.pool.tube_kmax,
                sample_mode=cfg.ddpm.sample_mode,
                edit_t0=cfg.ddpm.edit_t0,
                candidate_factor=cfg.ddpm.candidate_factor,
                realism_tv_gate=cfg.ddpm.realism_tv_gate,
                classic_al_oversample=getattr(cfg.pool, "classic_al_oversample", 10),
                sbal_alpha=getattr(cfg.pool, "sbal_alpha", 1.0),
            )
            finish_phase("make_pool", phase_start)
        else:
            print("round 0: using initial uniform pool", flush=True)
        print(f"round {round_id}: computing pretrain losses", flush=True)
        phase_start = time.perf_counter()
        pretrain_losses = compute_transition_losses(model, pool, cfg.surrogate.batch_size, device)
        pool.pretrain_losses = pretrain_losses
        if getattr(model, "n_models", 1) > 1:
            # Honest (out-of-sample) disagreement before the surrogate sees this pool;
            # used as the generator's difficulty label when difficulty_from=pretrain.
            pool.pretrain_uncertainty = compute_transition_uncertainty(
                model, pool, cfg.surrogate.batch_size, device
            )
        finish_phase("pretrain_losses", phase_start)
        # Garde-fou anti-collapse : on sauvegarde les poids AVANT l'entraînement du round.
        # Si le bulk val explose après (cf plus bas), on restaure ces poids. 0 = désactivé.
        pre_train_state = None
        if cfg.surrogate.degradation_guard > 0:
            pre_train_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"round {round_id}: training surrogate", flush=True)
        phase_start = time.perf_counter()
        train_metrics = train_surrogate(
            model=model,
            pool=pool,
            epochs=cfg.surrogate.epochs_per_round,
            batch_size=cfg.surrogate.batch_size,
            lr=cfg.surrogate.lr,
            weight_decay=cfg.surrogate.weight_decay,
            device=device,
            epoch_callback=make_epoch_callback(
                run, model, validation, cfg, device, round_id, "surrogate_epoch"
            ),
            round_id=round_id,
            seed=cfg.seed + round_id,
            ensemble_bootstrap=cfg.surrogate.ensemble_bootstrap,
            input_noise_std=cfg.surrogate.input_noise_std,
            loss_weighted_replay=cfg.surrogate.loss_weighted_replay,
            sobolev_weight=cfg.surrogate.sobolev_weight,
            pushforward_steps=getattr(cfg.surrogate, "pushforward_steps", 0),
            pushforward_weight=getattr(cfg.surrogate, "pushforward_weight", 0.5),
        )
        finish_phase("surrogate_train", phase_start)
        ddpm_metrics = {}
        if cfg.ddpm.enabled:
            print(f"round {round_id}: training ddpm", flush=True)
            phase_start = time.perf_counter()
            ddpm_metrics = train_ddpm(
                ddpm=ddpm,
                pool=pool,
                epochs=cfg.ddpm.train_epochs,
                batch_size=cfg.ddpm.batch_size,
                lr=cfg.ddpm.lr,
                device=device,
                state_mean=state_mean,
                state_std=state_std,
                pde=pde,
                mode=cfg.ddpm.mode,
                loss_metric=cfg.ddpm.loss_metric,
                difficulty_signal=cfg.ddpm.difficulty_signal,
                param_loss_weight=cfg.ddpm.param_loss_weight,
                loss_condition_scale=cfg.ddpm.loss_condition_scale,
                sample_weight_floor=cfg.ddpm.sample_weight_floor,
                sample_weight_power=cfg.ddpm.sample_weight_power,
                epoch_callback=make_epoch_callback(
                    run, model, validation, cfg, device, round_id, "ddpm_epoch"
                ),
                round_id=round_id,
                difficulty_from=cfg.ddpm.difficulty_from,
                difficulty_stats=cfg.ddpm.difficulty_stats,
                difficulty_stats_momentum=cfg.ddpm.difficulty_stats_momentum,
                self_train_weight=cfg.ddpm.self_train_weight,
                noise_prior=cfg.ddpm.noise_prior,
            )
            finish_phase("generator_train", phase_start)
        gen_eval_metrics: dict[str, float] = {}
        if cfg.ddpm.enabled and cfg.ddpm.mode == "conditional_quantile":
            print(f"round {round_id}: evaluating generator", flush=True)
            phase_start = time.perf_counter()
            ref_loss_mean = float(np.mean(pool.losses)) if pool.losses is not None else 1.0
            gen_eval_metrics = evaluate_generator(
                ddpm=ddpm,
                surrogate=model,
                pde=pde,
                val_ref_states=val_ref_states,
                ref_loss_mean=ref_loss_mean,
                state_mean=state_mean,
                state_std=state_std,
                rng=rng,
                device=device,
                loss_metric=cfg.ddpm.loss_metric,
                difficulty_signal=cfg.ddpm.difficulty_signal,
                n_eval=min(512, cfg.ddpm.sample_batch_size),
                solver_batch_size=cfg.ddpm.solver_batch_size,
                surrogate_batch_size=cfg.surrogate.batch_size,
            )
            finish_phase("generator_eval", phase_start)
            if gen_eval_metrics:
                print(json.dumps({k: round(v, 4) for k, v in gen_eval_metrics.items()}, sort_keys=True), flush=True)
        print(f"round {round_id}: full validation", flush=True)
        phase_start = time.perf_counter()
        eval_metrics = evaluate_validation(
            model=model,
            validation=validation,
            rollout_steps=cfg.validation.rollout_steps,
            quantiles=cfg.validation.quantiles,
            device=device,
        )
        finish_phase("full_validation", phase_start)
        # Garde-fou (suite) : si le bulk val a régressé de plus de `degradation_guard` x le
        # round précédent, on annule le round (restaure les poids d'avant) et on ré-évalue.
        guard_metrics: dict[str, float] = {}
        if pre_train_state is not None:
            guard_metrics["surrogate/guard_reverted"] = 0.0
            prev_bulk = history[-1].get("val/nrmse_mean") if history else None
            cur_bulk = eval_metrics.get("val/nrmse_mean")
            if (
                prev_bulk is not None
                and cur_bulk is not None
                and np.isfinite(prev_bulk)
                and cur_bulk > cfg.surrogate.degradation_guard * prev_bulk
            ):
                print(
                    f"round {round_id}: degradation guard fired "
                    f"(val/nrmse_mean {cur_bulk:.5f} > {cfg.surrogate.degradation_guard}x "
                    f"previous {prev_bulk:.5f}) — reverting surrogate to pre-round weights",
                    flush=True,
                )
                model.load_state_dict(pre_train_state)
                model.to(device)
                guard_metrics["surrogate/guard_reverted"] = 1.0
                guard_metrics["surrogate/guard_rejected_nrmse"] = float(cur_bulk)
                eval_metrics = evaluate_validation(
                    model=model,
                    validation=validation,
                    rollout_steps=cfg.validation.rollout_steps,
                    quantiles=cfg.validation.quantiles,
                    device=device,
                )
        hard_metrics: dict[str, float] = {}
        if hard_sets:
            phase_start = time.perf_counter()
            for name, hset in hard_sets.items():
                hard_metrics.update(
                    evaluate_one_step_validation(
                        model, hset, cfg.validation.quantiles, device, prefix=f"hard_val/{name}"
                    )
                )
            finish_phase("hard_validation", phase_start)
        timing_metrics["timing/round_before_diagnostics_sec"] = float(time.perf_counter() - round_start_time)
        metrics = {
            "round": round_id,
            "pool/n_samples": len(pool),
            "pool/loss_mean": float(np.mean(pool.losses)) if pool.losses is not None else 0.0,
            "pool/loss_p90": float(np.quantile(pool.losses, 0.9)) if pool.losses is not None else 0.0,
            **train_metrics,
            **ddpm_metrics,
            **gen_eval_metrics,
            **eval_metrics,
            **hard_metrics,
            **guard_metrics,
            **timing_metrics,
        }
        metrics.update(pool_debug_metrics(pool, pretrain_losses, cfg.ddpm.loss_metric))
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if run is not None:
            run.log(metrics)
            try:
                phase_start = time.perf_counter()
                log_wandb_diagnostics(
                    run, model, ddpm, pool, validation, cfg, device, round_id, round_id,
                    pde, state_mean, state_std,
                )
                diagnostics_sec = time.perf_counter() - phase_start
                print(f"round {round_id}: timing diagnostics={diagnostics_sec:.2f}s", flush=True)
                run.log({"round": round_id, "timing/diagnostics_sec": float(diagnostics_sec)})
            except Exception as exc:
                print(f"wandb diagnostics failed at round {round_id}: {exc}", file=sys.stderr)
        phase_start = time.perf_counter()
        pool_arrays = {
            "states": pool.states,
            "params": pool.params,
            "next_states": pool.next_states,
            "losses": pool.losses,
            "source": pool.source,
            "proposed_losses": pool.proposed_losses,
            "pretrain_losses": pool.pretrain_losses,
        }
        for key, value in (
            ("uncertainty", pool.uncertainty),
            ("pretrain_uncertainty", pool.pretrain_uncertainty),
            ("target_bins", pool.target_bins),
        ):
            if value is not None:
                pool_arrays[key] = value
        try:
            torch.save(model.state_dict(), out / "surrogate.pt", _use_new_zipfile_serialization=False)
        except Exception:
            torch.save(model.state_dict(), out / "surrogate.pt")
        try:
            torch.save(ddpm.state_dict(), out / "ddpm.pt", _use_new_zipfile_serialization=False)
        except Exception:
            torch.save(ddpm.state_dict(), out / "ddpm.pt")
        save_checkpoint(checkpoint, round_id, model, ddpm, pool, history, rng, wandb_run_id)
        
        # Continuously persist history.json at every round
        (out / "history.json").write_text(json.dumps(history, indent=2))
        
        save_sec = time.perf_counter() - phase_start
        round_total_sec = time.perf_counter() - round_start_time
        print(
            f"round {round_id}: timing save={save_sec:.2f}s total={round_total_sec:.2f}s",
            flush=True,
        )
        if run is not None:
            run.log(
                {
                    "round": round_id,
                    "timing/save_sec": float(save_sec),
                    "timing/round_total_sec": float(round_total_sec),
                }
            )

    (out / "history.json").write_text(json.dumps(history, indent=2))
    if run is not None:
        import os
        import signal
        import threading

        finished = threading.Event()

        def _finish():
            try:
                run.finish()
            finally:
                finished.set()

        t = threading.Thread(target=_finish, daemon=True)
        t.start()
        # Give wandb up to 3 minutes to upload; if it hangs, force-exit so the
        # OAR allocation is released rather than sitting idle on the GPU node.
        if not finished.wait(timeout=180):
            print("wandb.finish() timed out after 180s — forcing exit.", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
