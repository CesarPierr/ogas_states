"""Architecture/hyper-parameter sweep over generator configs."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from ._lg_scoring import score_generated_samples
from ._lg_training import train_loss_generator


SWEEP_CONFIGS = [
    # --- ODE steps ---
    {"name": "fm_s16",       "generator": "flow_matching", "hidden": 128, "steps": 16,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s32",       "generator": "flow_matching", "hidden": 128, "steps": 32,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s64",       "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s128",      "generator": "flow_matching", "hidden": 128, "steps": 128, "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    # --- Capacity (at s=64) ---
    {"name": "fm_s64_h64",   "generator": "flow_matching", "hidden": 64,  "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s64_h256",  "generator": "flow_matching", "hidden": 256, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10},
    # --- Residual blocks (at s=64, h=128) ---
    {"name": "fm_s64_rb0",   "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 0, "kernel_size": 7, "n_quantiles": 10},
    {"name": "fm_s64_rb4",   "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 4, "kernel_size": 7, "n_quantiles": 10},
    # --- n_quantiles (at s=64, h=128, rb=2) ---
    {"name": "fm_s64_nq5",   "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 5},
    {"name": "fm_s64_nq20",  "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 20},
    # --- quantile embed dim (at s=64, h=128, rb=2, nq=10; default = max(8, 128//8) = 16) ---
    {"name": "fm_s64_qed8",  "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10, "quant_embed_dim": 8},
    {"name": "fm_s64_qed32", "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10, "quant_embed_dim": 32},
    {"name": "fm_s64_qed64", "generator": "flow_matching", "hidden": 128, "steps": 64,  "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 10, "quant_embed_dim": 64},
    # --- production-candidate confirmation: best granularity (nq) at the lean hidden=64 backbone ---
    {"name": "fm_s64_h64_nq20", "generator": "flow_matching", "hidden": 64, "steps": 64, "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 20},
    {"name": "fm_s64_h64_nq5",  "generator": "flow_matching", "hidden": 64, "steps": 64, "residual_blocks": 2, "kernel_size": 7, "n_quantiles": 5},
]


def run_sweep(
    dataset_path: str | Path,
    output_dir: str | Path,
    configs: list[dict] | None = None,
    param_mode: str = "condition",
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 2e-4,
    n_quantiles: int = 10,
    seed: int = 0,
    device_name: str = "auto",
    n_score_samples: int = 1024,
    solver_batch_size: int = 512,
    jax_platform: str = "auto",
    skip_existing: bool = True,
    eval_every: int = 10,
    eval_n_samples: int = 512,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
) -> dict[str, dict]:
    if configs is None:
        configs = SWEEP_CONFIGS
    output_dir = Path(output_dir)
    summary: dict[str, dict] = {}
    for cfg in configs:
        name = cfg["name"]
        run_dir = output_dir / name
        # Resume: skip if score_metrics.json already exists
        if skip_existing and (run_dir / "score_metrics.json").exists():
            print(f"\nSkipping {name} (score_metrics.json exists)", flush=True)
            with open(run_dir / "score_metrics.json") as f:
                score = json.load(f)
            with open(run_dir / "history.json") as f:
                h = json.load(f)
            summary[name] = {**h[-1], **{k: v for k, v in score.items() if "top1/" in k or "uniform/" in k or "model/" in k or "reference/" in k}}
            continue
        print(f"\n{'='*60}\nSweep run: {name}\n{'='*60}", flush=True)
        t0 = time.perf_counter()
        # n_quantiles and quant_embed_dim can be overridden per-config entry
        cfg_nq = int(cfg.get("n_quantiles", n_quantiles))
        cfg_qed = int(cfg.get("quant_embed_dim", 0))
        ckpt_path, last_loss = train_loss_generator(
            dataset_path=dataset_path,
            output_dir=run_dir,
            generator=cfg.get("generator", "flow_matching"),
            param_mode=param_mode,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden=cfg.get("hidden", 128),
            steps=cfg.get("steps", 16),
            residual_blocks=cfg.get("residual_blocks", 2),
            kernel_size=cfg.get("kernel_size", 7),
            n_quantiles=cfg_nq,
            quant_embed_dim=cfg_qed,
            seed=seed,
            device_name=device_name,
            eval_every=eval_every,
            eval_n_samples=eval_n_samples,
            eval_solver_batch_size=solver_batch_size,
            eval_jax_platform=jax_platform,
            wandb_project=wandb_project,
            wandb_group=wandb_group or output_dir.name,
            wandb_run_name=name,
            wandb_extra_config={"sweep_name": output_dir.name, "config_name": name, **{k: cfg[k] for k in cfg if k != "name"}},
        )
        train_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        score = score_generated_samples(
            checkpoint_path=ckpt_path,
            dataset_path=dataset_path,
            output_dir=run_dir,
            n_samples=n_score_samples,
            seed=seed,
            device_name=device_name,
            solver_batch_size=solver_batch_size,
            jax_platform=jax_platform,
        )
        score_time = time.perf_counter() - t1
        summary[name] = {
            **last_loss,
            "train_time_s": train_time,
            "score_time_s": score_time,
            **{k: v for k, v in score.items() if "top1/" in k or "uniform/" in k or "model/" in k or "reference/" in k},
        }
        print(json.dumps({"run": name, **summary[name]}, sort_keys=True), flush=True)

    (output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    _print_sweep_table(summary, n_quantiles)
    return summary


def _print_sweep_table(summary: dict[str, dict], n_quantiles: int) -> None:
    """Print comparison table across all five evaluation axes."""
    W = 158
    print("\n" + "=" * W)
    hdr = (
        f"{'Model':<22}"
        f" {'params':>7} {'s':>4} {'nq':>3}"
        f" {'TV':>6} {'TV_r':>5}"          # realism: TV ratio
        f" {'amp_r':>6} {'psd_l2':>7}"     # realism: amplitude ratio, PSD distance
        f" {'top1_r':>7} {'uni_r':>6}"     # hardness: loss ratios
        f" {'cal_mae':>7} {'mono':>6} {'MI':>5}"  # conditioning fidelity
        f" {'ibd':>6}"                     # intra-bin diversity
        f" {'best_ep':>7} {'conv_ev':>7} {'conv_tr':>7}"  # convergence
        f" {'epoch_s':>7} {'ms/s':>6}"     # speed
    )
    print(hdr)
    print("-" * W)
    ref_tv = next(iter(summary.values())).get("reference_states/total_variation_mean", float("nan")) if summary else float("nan")
    val_tv = next(iter(summary.values())).get("validation_states/total_variation_mean", float("nan")) if summary else float("nan")
    print(f"  Reference (pool, uniform): TV={ref_tv:.1f}    Validation (real KS): TV={val_tv:.1f}")
    print("-" * W)
    for name, m in summary.items():
        tv = m.get("top1/states/total_variation_mean", float("nan"))
        tv_ref = val_tv if math.isfinite(val_tv) else ref_tv
        tv_ratio = tv / tv_ref if math.isfinite(tv) and tv_ref > 0 else float("nan")
        row = (
            f"{name:<22}"
            f" {m.get('model/n_params', 0):>7.0f}"
            f" {m.get('model/steps', 0):>4.0f}"
            f" {m.get('model/n_quantiles', 0):>3.0f}"
            # realism
            f" {tv:>6.1f}"
            f" {tv_ratio:>5.2f}"
            f" {m.get('top1/states/amplitude_ratio', float('nan')):>6.2f}"
            f" {m.get('top1/states/psd_log_l2', float('nan')):>7.2f}"
            # hardness
            f" {m.get('top1/loss_vs_reference_ratio', float('nan')):>7.2f}"
            f" {m.get('uniform/loss_vs_reference_ratio', float('nan')):>6.2f}"
            # conditioning fidelity (on top1 strategy)
            f" {m.get('top1/calibration_mae', float('nan')):>7.3f}"
            f" {m.get('top1/calibration_monotone_frac', float('nan')):>6.2f}"
            f" {m.get('top1/conditioning_mi_nats', float('nan')):>5.2f}"
            # diversity
            f" {m.get('top1/states/intra_bin_diversity_mean', float('nan')):>6.3f}"
            # convergence
            f" {m.get('best_eval_epoch', float('nan')):>7.0f}"
            f" {m.get('convergence_epoch_eval', float('nan')):>7.0f}"
            f" {m.get('convergence_epoch', float('nan')):>7.0f}"
            # speed
            f" {m.get('epoch_time_s', float('nan')):>7.2f}"
            f" {m.get('top1/gen_time_per_sample_ms', float('nan')):>6.2f}"
        )
        print(row)
    print("=" * W)
    print("Columns: TV=total-variation(top1), TV_r=TV/val_TV (1=perfect), amp_r=amplitude ratio (1=perfect),")
    print("         psd_l2=log-PSD L2 vs val (0=perfect), top1_r/uni_r=loss ratio vs reference (>1=harder),")
    print("         cal_mae=conditioning calibration err (0=perfect), mono=monotone frac (1=perfect),")
    print("         MI=mutual info nats (log(nq)=perfect), ibd=intra-bin diversity,")
    print("         best_ep=eval epoch with best combined score, conv_ev=first eval epoch within 5% of best,")
    print("         conv_tr=convergence epoch on train loss, epoch_s=mean s/epoch, ms/s=gen ms/sample")
