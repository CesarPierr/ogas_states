# WandB Metric Layout

This project logs on semantic x-axes rather than raw WandB `_step`, so uniform
and DDPM runs stay comparable even when DDPM runs emit extra training metrics.

## Axes

- `round`: active-learning round index. Used by round-level validation, pool,
  rollout, final training losses, figures, histograms, and tables.
- `surrogate_epoch/global_epoch`: continuous surrogate epoch index across all
  rounds. Used by `surrogate_epoch/*` and `surrogate_epoch_val/*`.
- `ddpm_epoch/global_epoch`: continuous DDPM epoch index across all rounds.
  Used by `ddpm_epoch/*`.

## Scalar Groups

- `train/*`: final surrogate training scalars per round.
- `ddpm/*`: final DDPM training scalars per round.
- `surrogate_epoch/*`: surrogate training loss and epoch counters.
- `surrogate_epoch_val/*`: one-step validation metrics logged every surrogate
  epoch on at most 10% of the fixed validation set.
- `ddpm_epoch/*`: DDPM denoising loss and epoch counters.
- `val/*`: one-step validation RMSE/NRMSE means, percentiles, and difficulty
  quantile splits on the fixed validation set.
- `rollout/*`: rollout RMSE/NRMSE means, final-step errors, and tail percentiles.
- `pool/*`: pool size, source counts, diversity, state quality, parameter
  summaries, loss summaries, and generated-sample correlations.
- `ddpm_sample/*`: scalar quality diagnostics for DDPM diagnostic samples.

## Non-Scalar Groups

- `pool_dist/*`: pool loss and parameter histograms.
- `ddpm_dist/*`: DDPM sample parameter histograms versus uniform references.
- `pool_figures/*`: sampled state heatmaps from the pool.
- `pool_figures/*/state_profiles`: readable 1D state profiles with a shared
  vertical scale.
- `val_figures/*`: one-step validation example panels.
- `rollout_figures/*`: rollout curves and squared spacetime panels combining
  ground truth, prediction, and absolute error. Rollout prediction and absolute
  error colors are scaled from the corresponding ground-truth q98 amplitude.
  RMSE/NRMSE rollout curves are split into mean, p95, and p99 plots with line
  keys based on the method-level WandB group, so seed overlays are easier to
  aggregate/read in the workspace.
- `ddpm_figures/*`: DDPM generated state heatmaps and 1D state profiles.
- `pool_tables/*`: compact sampled tables for parameters and losses.

## Main Dashboards To Watch

- Surrogate learning: `surrogate_epoch/epoch_loss`,
  `surrogate_epoch_val/nrmse_mean`, `surrogate_epoch_val/nrmse_p95`,
  `surrogate_epoch_val/nrmse_p99`.
- Round validation: `val/nrmse_mean`, `val/nrmse_p95`, `val/nrmse_p99`,
  `rollout/nrmse_mean`, `rollout/nrmse_final`,
  `rollout/nrmse_final_p99`.
- Pool behavior: `pool/loss_mean`, `pool/loss_p90`,
  `pool/generated_pretrain_loss_mean`, `pool/uniform_pretrain_loss_mean`,
  `pool/generated_pairwise_l2_mean`,
  `pool/generated_proposed_vs_pretrain_loss_corr`.
- DDPM health: `ddpm_epoch/epoch_loss`, `ddpm/loss`,
  `ddpm_sample/states/finite_ratio`, `ddpm_sample/states/abs_p99`,
  `ddpm_sample/states/highfreq_energy_ratio_mean`.
