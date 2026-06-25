"""Loss-generator lab: standalone offline tool to build a (state, difficulty) dataset
from a finished run, train/sweep loss-conditioned generators, and score them against
real-KS statistics. NOT used by the live training loop (run.py). Implementation lives
split across the sub-package modules; this file keeps the CLIs and public API.
"""
from __future__ import annotations

import argparse

from .common import _dataclass_from_dict, assign_loss_quantile_bins, choose_device, config_from_resolved_dict, load_resolved_config, loss_norm_constants, normalize_losses_with, stratified_split
from .dataset import _recompute_losses_with_surrogate, _validation_state_stats, build_loss_dataset_from_run
from .generator import _balanced_sampler_from_quantiles, _sampling_strategies, _scale_params, _softmax, _unscale_params, accepted_model_kwargs, make_generator, sample_loss_generator
from .scoring import _load_validation_ref_states, _pde_step_batch, score_generated_samples
from .sweep import SWEEP_CONFIGS, _print_sweep_table, run_sweep
from .training import _summarize_eval_for_history, train_loss_generator

__all__ = [
    'SWEEP_CONFIGS',
    '_balanced_sampler_from_quantiles',
    '_dataclass_from_dict',
    '_load_validation_ref_states',
    '_pde_step_batch',
    '_print_sweep_table',
    '_recompute_losses_with_surrogate',
    '_sampling_strategies',
    '_scale_params',
    '_softmax',
    '_summarize_eval_for_history',
    '_unscale_params',
    '_validation_state_stats',
    'accepted_model_kwargs',
    'assign_loss_quantile_bins',
    'build_dataset_cli',
    'build_loss_dataset_from_run',
    'choose_device',
    'config_from_resolved_dict',
    'load_resolved_config',
    'loss_norm_constants',
    'make_generator',
    'normalize_losses_with',
    'run_sweep',
    'sample_loss_generator',
    'score_generated_samples',
    'stratified_split',
    'sweep_cli',
    'train_cli',
    'train_loss_generator',
    'build_dataset_cli',
    'train_cli',
    'sweep_cli',
]


def build_dataset_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--n-quantile-bins", type=int, default=10)
    parser.add_argument("--recompute-with-final-surrogate", action="store_true",
                        help="Re-evaluate pool losses using surrogate.pt instead of the per-round losses")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    path = build_loss_dataset_from_run(
        args.run_dir,
        output_path=args.output,
        round_id=args.round,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        n_quantile_bins=args.n_quantile_bins,
        recompute_with_final_surrogate=args.recompute_with_final_surrogate,
        device_name=args.device,
    )
    print(path)


def train_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generator", choices=["ddpm", "flow_matching"], default="flow_matching")
    parser.add_argument("--param-mode", choices=["condition", "generate"], default="condition")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--loss-metric", choices=["mse", "rmse"], default="rmse")
    parser.add_argument("--n-quantiles", type=int, default=10)
    parser.add_argument("--quant-embed-dim", type=int, default=0)
    # Conditioning-HPO knobs, forwarded to the generator constructor only when set
    # (so current models keep working until the extended DDPMConfig branch merges).
    parser.add_argument("--cfg-dropout", type=float, default=None,
                        help="Classifier-free-guidance conditioning dropout (only forwarded when set)")
    parser.add_argument("--cfg-scale", type=float, default=None,
                        help="Classifier-free-guidance scale at sampling (only forwarded when set)")
    parser.add_argument("--cond-mode", choices=["concat", "film"], default=None,
                        help="Conditioning injection mode (only forwarded when set)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jax-platform", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--score-samples", type=int, default=1024)
    parser.add_argument("--solver-batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run full 5-axis validation eval every N epochs (0 = disable)")
    parser.add_argument("--eval-n-samples", type=int, default=512)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args(argv)
    model_kwargs = {
        k: v
        for k, v in {"cfg_dropout": args.cfg_dropout, "cfg_scale": args.cfg_scale, "cond_mode": args.cond_mode}.items()
        if v is not None
    }
    ckpt, _ = train_loss_generator(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        generator=args.generator,
        param_mode=args.param_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        steps=args.steps,
        residual_blocks=args.residual_blocks,
        kernel_size=args.kernel_size,
        loss_metric=args.loss_metric,
        n_quantiles=args.n_quantiles,
        quant_embed_dim=args.quant_embed_dim,
        model_kwargs=model_kwargs,
        seed=args.seed,
        device_name=args.device,
        eval_every=args.eval_every,
        eval_n_samples=args.eval_n_samples,
        eval_solver_batch_size=args.solver_batch_size,
        eval_jax_platform=args.jax_platform,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
    )
    if args.score_samples > 0:
        score_generated_samples(
            ckpt,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            n_samples=args.score_samples,
            seed=args.seed,
            device_name=args.device,
            solver_batch_size=args.solver_batch_size,
            jax_platform=args.jax_platform,
        )


def sweep_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--param-mode", choices=["condition", "generate"], default="condition")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-quantiles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-samples", type=int, default=1024)
    parser.add_argument("--solver-batch-size", type=int, default=512)
    parser.add_argument("--jax-platform", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Run full 5-axis validation eval every N epochs (0 = disable)")
    parser.add_argument("--eval-n-samples", type=int, default=512,
                        help="Samples per strategy at intermediate evals (final score uses --score-samples)")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project for per-config runs (omit to disable)")
    parser.add_argument("--wandb-group", default=None,
                        help="W&B group (default: sweep output dir name)")
    args = parser.parse_args(argv)
    run_sweep(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        param_mode=args.param_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_quantiles=args.n_quantiles,
        seed=args.seed,
        device_name=args.device,
        n_score_samples=args.score_samples,
        solver_batch_size=args.solver_batch_size,
        eval_every=args.eval_every,
        eval_n_samples=args.eval_n_samples,
        jax_platform=args.jax_platform,
        skip_existing=not args.no_skip_existing,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
    )
