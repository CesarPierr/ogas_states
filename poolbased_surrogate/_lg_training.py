"""Train one loss-conditioned generator and track its eval history."""
from __future__ import annotations

import json
import math
import numpy as np
import os
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from typing import Any
from .train import normalize_losses, transform_transition_losses
from ._lg_common import assign_loss_quantile_bins, choose_device
from ._lg_generator import _balanced_sampler_from_quantiles, _scale_params, make_generator
from ._lg_scoring import score_generated_samples


def _summarize_eval_for_history(score: dict[str, float], epoch: int, eval_time_s: float) -> dict[str, float]:
    """Reduce a full score_metrics dict to a compact per-eval-epoch summary.

    Captures one row per axis so we can plot how realism / hardness / conditioning /
    diversity / speed evolve over training, and derive a single combined_score that
    drives convergence-epoch detection.
    """
    def _f(key: str, default: float = float("nan")) -> float:
        v = score.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    ref_tv = _f("validation_states/total_variation_mean", _f("reference_states/total_variation_mean", float("nan")))
    tv_top1 = _f("top1/states/total_variation_mean")
    tv_ratio = tv_top1 / ref_tv if math.isfinite(tv_top1) and math.isfinite(ref_tv) and ref_tv > 0 else float("nan")

    # Combined score: realism × hardness × conditioning. See analyze_sweep.recommend().
    if math.isfinite(tv_ratio) and tv_ratio <= 3.0:
        tv_score = math.exp(-0.5 * (tv_ratio - 1.0) ** 2)
    else:
        tv_score = 0.0
    loss_ratio_top1 = _f("top1/loss_vs_reference_ratio")
    ratio_norm = min(loss_ratio_top1 / 5.0, 1.0) if math.isfinite(loss_ratio_top1) else 0.0
    calib_mae = _f("top1/calibration_mae")
    calib_score = max(0.0, 1.0 - calib_mae) if math.isfinite(calib_mae) else 0.0
    combined = 0.4 * tv_score + 0.4 * ratio_norm + 0.2 * calib_score

    return {
        "epoch": float(epoch),
        "eval_time_s": float(eval_time_s),
        # realism
        "tv_top1": tv_top1,
        "tv_ratio": tv_ratio,
        "amplitude_ratio_top1": _f("top1/states/amplitude_ratio"),
        "psd_log_l2_top1": _f("top1/states/psd_log_l2"),
        # hardness
        "loss_ratio_top1": loss_ratio_top1,
        "loss_ratio_uniform": _f("uniform/loss_vs_reference_ratio"),
        # conditioning
        "calibration_mae_top1": calib_mae,
        "calibration_monotone_frac_top1": _f("top1/calibration_monotone_frac"),
        "conditioning_mi_uniform": _f("uniform/conditioning_mi_nats"),
        "rank_corr_uniform": _f("uniform/target_vs_realized_rank_corr"),
        # diversity
        "intra_bin_diversity_top1": _f("top1/states/intra_bin_diversity_mean"),
        # speed
        "gen_time_ms_top1": _f("top1/gen_time_per_sample_ms"),
        # combined
        "tv_score": tv_score,
        "ratio_norm": ratio_norm,
        "calib_score": calib_score,
        "combined_score": combined,
    }


def train_loss_generator(
    dataset_path: str | Path,
    output_dir: str | Path,
    generator: str = "flow_matching",
    param_mode: str = "condition",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 2e-4,
    hidden: int = 128,
    steps: int = 16,
    residual_blocks: int = 2,
    kernel_size: int = 7,
    loss_metric: str = "rmse",
    n_quantiles: int = 10,
    quant_embed_dim: int = 0,
    seed: int = 0,
    device_name: str = "auto",
    eval_every: int = 0,
    eval_n_samples: int = 512,
    eval_solver_batch_size: int = 512,
    eval_jax_platform: str = "auto",
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    wandb_extra_config: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, float]]:
    """Train a state generator conditioned on quantile labels.

    States are z-normalised before training (mean/std from validation or pool)
    and denormalised back to physical units at sampling time.

    The conditioning is a discrete quantile label in {0, …, n_quantiles-1}
    routed through a learned embedding.  Training uses a balanced per-bin
    sampler so the model sees equal signal from every difficulty level.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device(device_name)

    with np.load(dataset_path, allow_pickle=False) as data:
        states = data["states"].astype(np.float32)
        params = data["params"].astype(np.float32)
        raw_losses = data["losses"].astype(np.float32).reshape(-1)
        split = data["split"].astype(np.int8)
        state_mean = float(data["state_mean"])
        state_std = float(data["state_std"])
        param_min = data["param_min"].astype(np.float32)
        param_max = data["param_max"].astype(np.float32)
        resolved_config = str(data["resolved_config"])
        # Re-assign quantile bins from the full dataset so edges are consistent,
        # unless n_quantiles is overridden/different.
        if "quantile_edges" in data and (len(data["quantile_edges"]) - 1) == n_quantiles:
            quantile_edges = data["quantile_edges"].astype(np.float32)
            quantile_bins_all = data["loss_quantile_bins"].astype(np.int8)
        else:
            quantile_bins_all, quantile_edges = assign_loss_quantile_bins(
                normalize_losses(transform_transition_losses(raw_losses, loss_metric)), n_quantiles
            )

    train_mask = split == 0
    # States are z-normalised: the generator trains in a unit-Gaussian-like space,
    # making it much easier to learn the diffusion/flow objective.
    state_norm = ((states[train_mask] - state_mean) / state_std).astype(np.float32)
    params_scaled = _scale_params(params[train_mask], param_min, param_max)
    quantile_train = quantile_bins_all[train_mask].astype(np.int64)

    dataset = TensorDataset(
        torch.from_numpy(state_norm),
        torch.from_numpy(quantile_train),
        torch.from_numpy(params_scaled),
    )
    # Balanced sampler: equal bin frequency so conditioning signal has uniform coverage.
    sampler = _balanced_sampler_from_quantiles(quantile_train, n_quantiles)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)

    model = make_generator(
        generator=generator,
        resolution=states.shape[-1],
        hidden=hidden,
        steps=steps,
        loss_conditional=True,
        param_dim=params.shape[1],
        param_mode=param_mode,
        residual_blocks=residual_blocks,
        kernel_size=kernel_size,
        n_quantiles=n_quantiles,
        quant_embed_dim=quant_embed_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []
    n_params = sum(p.numel() for p in model.parameters())
    train_start = time.perf_counter()

    # Optional wandb run for this single config
    wandb_run = None
    if wandb_project:
        try:
            import wandb  # noqa: PLC0415
            wandb_cfg = {
                "generator": generator,
                "param_mode": param_mode,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "hidden": hidden,
                "steps": steps,
                "residual_blocks": residual_blocks,
                "kernel_size": kernel_size,
                "loss_metric": loss_metric,
                "n_quantiles": n_quantiles,
                "quant_embed_dim": quant_embed_dim,
                "seed": seed,
                "n_params": n_params,
                "dataset_path": str(dataset_path),
                "eval_every": eval_every,
                "eval_n_samples": eval_n_samples,
            }
            if wandb_extra_config:
                wandb_cfg.update(wandb_extra_config)
            init_kwargs = {
                "project": wandb_project,
                "name": wandb_run_name,
                "group": wandb_group,
                "config": wandb_cfg,
                "dir": str(output_dir),
                "reinit": True,
            }
            if not os.environ.get("WANDB_API_KEY"):
                init_kwargs["anonymous"] = "allow"
            wandb_run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
            wandb_run.define_metric("epoch")
            wandb_run.define_metric("train/*", step_metric="epoch")
            wandb_run.define_metric("eval/*", step_metric="epoch")
        except Exception as e:  # noqa: BLE001
            print(f"  [wandb] init failed ({e}) — continuing without wandb", flush=True)
            wandb_run = None

    def _build_ckpt() -> dict:
        return {
            "model": model.state_dict(),
            "dataset_path": str(dataset_path),
            "resolved_config": resolved_config,
            "generator": generator,
            "param_mode": param_mode,
            "hidden": hidden,
            "steps": steps,
            "residual_blocks": residual_blocks,
            "kernel_size": kernel_size,
            "loss_metric": loss_metric,
            "n_quantiles": n_quantiles,
            "quant_embed_dim": quant_embed_dim,
            "quantile_edges": quantile_edges,
            "state_mean": state_mean,
            "state_std": state_std,
            "param_min": param_min,
            "param_max": param_max,
            "n_params": n_params,
        }

    ckpt_path = output_dir / "loss_generator.pt"

    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        model.train()
        losses = []
        for state_b, qbin_b, param_b in loader:
            opt.zero_grad(set_to_none=True)
            loss = model.training_loss(
                state_b.to(device),
                params=param_b.to(device) if model.use_param_cond else None,
                quantile_label=qbin_b.to(device),
                param_loss_weight=1.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        record = {
            "epoch": float(epoch),
            "train/loss": float(np.mean(losses)),
            "epoch_time_s": float(time.perf_counter() - epoch_t0),
            "wall_time_s": float(time.perf_counter() - train_start),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": record["train/loss"],
                "train/epoch_time_s": record["epoch_time_s"],
                "train/wall_time_s": record["wall_time_s"],
            })

        # Periodic full 5-axis validation eval
        do_eval = (
            eval_every > 0
            and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
        )
        if do_eval:
            eval_t0 = time.perf_counter()
            # Persist current model so score_generated_samples can load it.
            torch.save(_build_ckpt(), ckpt_path)
            eval_dir = output_dir / f"eval_ep{epoch + 1}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [eval@epoch={epoch + 1}] scoring (n_samples={eval_n_samples}) ...", flush=True)
            try:
                score = score_generated_samples(
                    checkpoint_path=ckpt_path,
                    dataset_path=dataset_path,
                    output_dir=eval_dir,
                    n_samples=eval_n_samples,
                    seed=seed,
                    device_name=device_name,
                    solver_batch_size=eval_solver_batch_size,
                    jax_platform=eval_jax_platform,
                )
                model.train()  # score_generated_samples puts model into eval mode through sampling
            except Exception as e:  # noqa: BLE001
                print(f"  [eval@epoch={epoch + 1}] FAILED: {e}", flush=True)
                score = {}
            eval_time = time.perf_counter() - eval_t0
            eval_entry = _summarize_eval_for_history(score, epoch=epoch + 1, eval_time_s=eval_time)
            eval_history.append(eval_entry)
            # Surface eval entry in history.json so wall-clock plots include it
            history[-1] = {**record, **{f"eval/{k}": v for k, v in eval_entry.items() if k != "epoch"}}
            print(json.dumps({"eval_summary": eval_entry}), flush=True)
            (output_dir / "eval_history.json").write_text(json.dumps(eval_history, indent=2))
            if wandb_run is not None:
                # Log the compact high-level summary metrics under eval/
                wandb_payload = {"epoch": epoch}
                for k, v in eval_entry.items():
                    if k == "epoch":
                        continue
                    try:
                        wandb_payload[f"eval/{k}"] = float(v)
                    except (TypeError, ValueError):
                        pass
                
                # Log selected key full metrics for top1 and uniform strategies to avoid metric clutter
                for k, v in score.items():
                    # Only keep top1 and uniform strategy metrics, and essential reference stats
                    if not (k.startswith("top1/") or k.startswith("uniform/")):
                        if k.startswith("reference_states/") or k.startswith("validation_states/") or k.startswith("reference/"):
                            if not any(p in k for p in ["total_variation_mean", "loss_mean"]):
                                continue
                        else:
                            continue
                    
                    # Filter out highly detailed, noisy, or redundant statistics
                    exclude_patterns = [
                        "per_bin", "states/mean", "states/std", "states/min", "states/max",
                        "states/abs_p", "states/pairwise_l2_", "states/highfreq_energy", "states/roughness",
                        "value_mean", "value_std", "target_quantile_mean", "realized_quantile_mean",
                        "exact_quantile_accuracy", "top_quantile_hit_rate"
                    ]
                    if any(p in k for p in exclude_patterns):
                        continue
                        
                    try:
                        # Log under eval/ to keep namespace shallow and clean
                        wandb_payload[f"eval/{k}"] = float(v)
                    except (TypeError, ValueError):
                        pass
                wandb_run.log(wandb_payload)

    total_train_time = time.perf_counter() - train_start

    # Convergence epoch (train loss): first epoch where loss ≤ 1.05× min(loss)
    all_losses = [h["train/loss"] for h in history]
    min_loss = min(all_losses)
    convergence_epoch = next((i for i, l in enumerate(all_losses) if l <= 1.05 * min_loss), epochs - 1)

    # Convergence epoch (validation): first eval-checkpoint whose combined_score
    # is within 5% of the best across all eval points.
    best_eval_epoch = -1
    convergence_epoch_eval = -1
    if eval_history:
        scored = [(int(e["epoch"]), float(e.get("combined_score", float("nan")))) for e in eval_history]
        scored = [(ep, s) for ep, s in scored if math.isfinite(s)]
        if scored:
            best_eval_epoch, best_score = max(scored, key=lambda x: x[1])
            threshold = 0.95 * best_score if best_score > 0 else best_score
            convergence_epoch_eval = next((ep for ep, s in scored if s >= threshold), best_eval_epoch)

    ckpt = {
        **_build_ckpt(),
        "history": history,
        "eval_history": eval_history,
        "total_train_time_s": total_train_time,
        "convergence_epoch": convergence_epoch,
        "convergence_epoch_eval": convergence_epoch_eval,
        "best_eval_epoch": best_eval_epoch,
    }
    torch.save(ckpt, ckpt_path)
    last = {
        **history[-1],
        "total_train_time_s": total_train_time,
        "convergence_epoch": float(convergence_epoch),
        "convergence_epoch_eval": float(convergence_epoch_eval),
        "best_eval_epoch": float(best_eval_epoch),
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    if eval_history:
        (output_dir / "eval_history.json").write_text(json.dumps(eval_history, indent=2))
    if wandb_run is not None:
        try:
            for k, v in last.items():
                try:
                    wandb_run.summary[f"final/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
            wandb_run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"  [wandb] finish failed: {e}", flush=True)
    return ckpt_path, last
