# Project Handoff — Generative-State Active Sampling for PDE Surrogates

State snapshot (2026-06-12) for restarting work on any machine. Companion: `LEONARDO.md` (migration).

## Thesis
Train one-step neural PDE surrogates by **generating difficulty-targeted training states**
(flow-matching generator, solver-labeled) mixed with uniform attractor trajectories
(`uniform_fraction` α). Goal: low error on rare/hard states and off-attractor "tube" states
⇒ stable rollouts — at near-neutral bulk cost. Novelty check (8 lit sweeps): nobody does
difficulty-conditioned generative state proposal with solver labeling; AL4PDE & co. select
ICs/params only.

## What is empirically established (Phase 2 clean, KS res800, 5 seeds, round 10, v2 analysis)
- **Robust**: tube one-step RMSE −54…−79% (monotone in ρ); TV-hard p99 −60…−66%; uf00 bulk +209%.
- **Not robust** (flipped or within seed noise at n=5): "ensvar bulk-neutral" (v2: +4.4%),
  rollout@100 gains (ensvar seed303 blows up: ~+69%), K_τ "9–29×" (metric floor-saturated;
  only p95@τ=2.0 column moves), ensvar>loss signal (null on Burgers).
- **Adverse, must be reported**: lowamp-hard worse for ALL variants (+18…+189%); sup/mean &
  Gini worse on tube sets; per-sample conditioning correlation ≈ 0 (gains were distribution-level).
- **Burgers Phase 3 eval invalid as built**: 31–47% of the validation bank blows up into ±5
  sawtooth solver artifacts (high ν); TV-hard sets 100% artifact; bank must be regenerated with
  a stability gate. Runs themselves completed fine (30/30).
- Paper table (paper/main.tex, uncommitted edits) used the stale 4-seed v1 analysis — paper is
  to be REWRITTEN from scratch; treat current main.tex as scrap.

## The v3 method (implemented 2026-06-12, committed; smoke-tested; pilot launched)
Diagnosed failure → fix (all config-switchable, see `configs/bigfoot_ks_v3_base.yaml`):
- overfit post-train labels → `ddpm.difficulty_from: pretrain`
- drifting bin semantics → `ddpm.difficulty_stats: ema` (buffer-backed, checkpointed)
- generator self-feeding → `ddpm.self_train_weight: 0.3`; spectrum prior fitted on uniform-sourced states only
- white-noise roughness → `ddpm.noise_prior: spectrum` (coloured base in BOTH training & sampling)
- non-functional steering → **oversample-then-reject**: `ddpm.candidate_factor: 4` + TV gate
  (`realism_tv_gate: 3.0`) + keep top by ensemble disagreement (zero extra solver calls)
- init-only ensemble diversity → per-member shuffled loaders + bootstrap (`surrogate.ensemble_bootstrap`)
- no manifold prior → `ddpm.sample_mode: edit` (SDEdit around attractor anchors, `edit_t0: 0.6`)
- audit gaps → `target_bins`/`uncertainty`/`pretrain_uncertainty` persisted in pool npz; fallback states `source=2`
- smoke result: generated TV 22 vs uniform 40 (was 3–10× rougher before) — realism fixed.

## Pilot (6 arms × seed 101 × 10 rounds) — Bigfoot OAR 10910-10915, queued
| arm | tests |
|---|---|
| uniform_baseline | control |
| noise_inject (σ=0.25) | published zero-cost robustness baseline |
| random_tube | anchors + random low-k noise, NO learning — generator-necessity test |
| mined_ic | disagreement-selected random ICs — selection-without-generation |
| gen_v3 | full v3 generator (scratch) |
| gen_v3_edit | v3 in SDEdit mode |
Decision: gen_v3* must beat random_tube on the NEW diverse suite, else the generator is unjustified.
Eval on `ks_diverse_suite` (diversity-constrained hard sets + 3-band tube low/mid/high-k ×
ρ∈{0.1,0.25,0.5}; built by `scripts/build_diverse_validation.py`).

## Theory pillars for the rewrite (verified citations; from the lit-search workflow)
1. **Fill distance ⇒ sup-norm error, intrinsic dimension**: Wendland Thm 11.1; Narcowich–Ward–Wendland
   (NN-applicable); Fuselier–Wright (manifold rates); KS inertial manifold dim O(10) (Foias/Nicolaenko/
   Temam lineage); Köhne et al. 2024 (L∞ one-step surrogate ≤ C·h^k, arbitrary training points);
   Reznikov–Saff (trajectory sampling ⇒ fill distance dominated by rarest region = why uniform fails).
   ⇒ sampler objective = minimize fill distance over attractor+tube.
2. **α-mixture certificate**: defensive IS (Hesterberg; Owen–Zhou) dq/dp ≤ 1/α; Cortes–Mansour–Mohri
   transfer bound; He–Owen: α=1/2 is minimax-regret optimal for 2 components.
3. **CVaR-optimal sampling**: Rockafellar–Uryasev dual ⇒ optimal sampler = current-model tail-conditional;
   Ada-CVaR (lagging sampler converges); Levy et al. 1/β floor for p-only sampling ⇒ synthesis provably
   beats reweighting for tail risk; elite-fitting = cross-entropy method (collapse prevented by α-mix).
4. **Honest expectations**: Chen–Price (bulk gains capped at constant factor — neutral bulk is predicted);
   Castro–Willett–Nowak (gains where difficulty concentrates on lower-dim sets; optimal algo = uniform
   preview + targeted refinement = α-mixture).
5. **2D/latent (parked)**: coverage transfers through bi-Lipschitz decoder (BLAE 2026 machinery; latent
   dim ~2× intrinsic); diffusion sample complexity linear in intrinsic dim even ambient (Potaptchik et al.)
   ⇒ latent sampling is a compute argument, not statistics.
Full results: theory workflow output (8 areas × ~13 guarantees, 44/56 verified) — see session transcript.

## Immediate next steps
1. Pilot results → if gen_v3* > random_tube on diverse tube/hard: 5-seed campaign of the winning arms.
2. Conditioning fidelity now measurable (target_bins persisted): rank-corr(target bin, realized
   disagreement) per round.
3. Regenerate Burgers bank with stability gate before any Burgers claim.
4. lowamp-hard fix (second conditioning head on target amplitude) if pilot confirms regression persists.
5. Paper rewrite on pillars 1–4; current main.tex is scrap.

## Known open issues
- `loss_generator_lab` scoring units bug fixed only for finite-guard; per-batch normalization vs pool
  edges mismatch still pending in `_lg_scoring.py:201` (lab-only, not production).
- Old checkpoints (pre-v3) can't be resumed by v3 code (new model buffers) — fresh runs only.
- al4pde Burgers solver: batch-size-1 serial workaround in `al4pde_bridge.py` (JAX shape bug upstream).
