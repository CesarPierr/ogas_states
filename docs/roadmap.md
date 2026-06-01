# Plan global de recherche & exécution — surrogate PDE par génération d'états durs

> **Objectif :** un papier A* (NeurIPS / ICML / ICLR). Ce document agrège **toutes** les idées
> accumulées, les ordonne en une thèse unique, et définit un plan d'exécution par phases avec
> jalons go/no-go pour ne pas se disperser. Théorie détaillée : [theory.md](theory.md).

---

## 0. La thèse en une phrase

> *Pour les surrogates neuronaux de PDE, l'échantillonnage de trajectoires uniformes optimise
> la mauvaise fonctionnelle (la moyenne sur l'attracteur) et souffre d'une limite de
> **couverture** ; en **générant et recherchant les états durs** on optimise un objectif de
> **couverture / robustesse (DRO)** — uniformité de la loss sur tout l'espace atteignable — ce
> qui (i) est mesurable seulement sur une validation dure/non-vue, et (ii) fournit une
> **garantie de stabilité du rollout** via la couverture du tube (anti-exposure-bias).*

Trois piliers, chacun = une contribution :
1. **Reformulation** : objectif coverage/DRO + « uniformité sur la loss » ([theory.md §7](theory.md)).
2. **Garantie rollout** : couverture du tube ⇒ invariance avant ⇒ rollout borné / horizon
   allongé ; « DAgger anticipé » par modèle génératif ([theory.md §9](theory.md)).
3. **Méthode + mesure** : générateur conditionné par difficulté (bins de quantile, lois de
   sampling), `uniform_fraction` = rayon DRO borné (`R_p ≤ R_q/α`, [§3.1](theory.md)), signal
   = désaccord d'ensemble ([§3.3](theory.md)), **validation dure/non-vue** ([§8](theory.md)).

---

## 1. Inventaire des idées (et leur statut)

| # | Idée | Rôle dans la thèse | Statut |
|---|------|--------------------|--------|
| I1 | Générateur d'états durs (flow/ddpm) conditionné difficulté | méthode centrale | ✅ implémenté (`fm_s64_h64`, nq20) |
| I2 | Bins de quantile + lois de sampling (top_half, exp_bias…) | mécanisme de ciblage | ✅ implémenté |
| I3 | `uniform_fraction` = mélange `αp+(1−α)q_gen` | bouton DRO, borne `R_p≤R_q/α` | ✅ ; balayage à faire |
| I4 | Signal difficulté = **désaccord d'ensemble** | épistémique réductible (§3.3) | ✅ implémenté (ce commit) ; à tester |
| I5 | Conditionnement NRMSE | — | ❌ **negative result** (chasse états triviaux) |
| I6 | Reformulation **coverage / DRO / loss-uniform** | pilier 1 | ✅ formalisé (§7) |
| I7 | **Validation dure / non-vue** (TV, low-amp, params, tube) | mesure du gain | 🟡 builder+post-hoc OK ; sets à étendre |
| I8 | **Stabilité rollout = couverture du tube** | pilier 2, garantie | ✅ formalisé (§9) ; à tester |
| I9 | **Rollout-aware hard mining** (boucle fermée / DAgger) | variante forte rollout | 🔲 à implémenter |
| I10 | Figure **efficacité-solveur** (3 baselines) | data-efficiency | 🔲 à implémenter |
| I11 | Surrogate plus fort (hidden64, ensemble) | sortir du régime dégénéré | 🔲 config |
| I12 | **Multi-PDE** (1D suite → 2D) | généralité (exigence A*) | 🔲 phase 3 |

Légende : ✅ fait · 🟡 partiel · 🔲 à faire · ❌ écarté (à documenter).

---

## 2. Découpage en phases (avec jalons go/no-go)

### Phase 0 — Cadrage théorique  *(fait)*
- [theory.md](theory.md) §1–§9 : covariate-shift, borne de mélange, fonctionnelle rollout,
  reformulation coverage/DRO, validation, tube-invariance.
- **Livrable :** énoncés + 2 propositions (borne `R_p≤R_q/α` ; tube piégeant ⇒ rollout borné).

### Phase 1 — La métrique d'abord : prouver que le gain est *masqué*, pas absent
*Hypothèse H-mesure (§8.5) : le bénéfice du hard-mining est grand sur validation dure/non-vue,
faible sur uniforme.*
- **P1.1** Sets de validation durs model-independent : `val_hard_tv` (RMSE-dur), `val_hard_lowamp`
  (NRMSE-dur), `val_cov` (loss-uniform). ✅ `scripts/build_hard_validation.py`.
- **P1.2** Eval **post-hoc** sur les runs existants (uniforme vs variantes) :
  `scripts/eval_hard_posthoc.py` → NRMSE + inégalité du profil (sup/mean, Gini) par set.
- **P1.3** Set **params non-vus** (§8.1) : nécessite `param_train_ranges ⊊ param_ranges`
  (training restreint) → nouveaux runs.
- **Jalon G1 :** si (variante − uniforme) est nettement plus négatif sur `val_hard`/`val_cov`
  que sur `val_unif`, **GO** : la « demi-teinte » était un artefact de métrique. Sinon →
  revoir réalisme/diversité/alignement du générateur (§7.4) avant d'investir plus.

### Phase 2 — La méthode forte : couverture du tube & rollout
- **P2.1** Surrogate hidden64 + ensemble (I11/I4) pour rollout non-dégénéré (`nrmse≪1`).
- **P2.2** Balayage `uniform_fraction ∈ {0,.25,.5,.75,1}` (I3) → **front de Pareto**
  bulk vs queue/rollout ; vérifier la borne `R_p/R_q ≤ 1/α`.
- **P2.3** Signal difficulté : **loss vs désaccord d'ensemble** (I4) à `α` optimal.
- **P2.4** **Tube de validation** (§8.4/§9.5) : états d'attracteur perturbés à l'échelle ρ ;
  mesurer `ε̄` (couverture), horizon stable `K_τ`, `dist(û_k, M_traj)` le long du rollout.
- **P2.5** **Rollout-aware hard mining** (I9) : difficulté = erreur sur le tube visité /
  seeds perturbés ; comparer au hard-mining one-step.
- **Jalon G2 :** la méthode allonge l'horizon stable `K_τ` et borne `dist(û_k,M)` vs uniforme
  (test direct de §9). **GO** vers la rédaction si oui.
- **Statut (2026-06-01) :** campagne lancée — 6 variantes (`uf∈{0,.25,.5,.75}` + `uniform_baseline`
  + `uf50_ensvar`) × 5 seeds, `ks800_phase2_5seed`. **Budget = 10 rounds** (la val uniforme sature dès
  le round ~3 ; 10 rounds capturent la tendance couverture/tube et tiennent dans le walltime V100 sans
  relance A100). **Pool à budget fixe redistribué** (17 500 transitions/round, non cumulé) ⇒ « round » =
  proxy de l'axe d'efficacité (nb d'appels solveur) ; comparaison **à round 10 apparié** = appels-solveur
  appariés. Analyse via `eval_hard_posthoc.py` (Pareto + tube) et `eval_rollout_posthoc.py` (`K_τ`).
  GPU : A100 rapide / V100 ~3.3× plus lent sur surrogate+validation (TF32 vs FP32), neutre sur le
  générateur (overhead-bound).

### Phase 3 — Généralité multi-PDE  *(exigence A*)*
- **P3.1 (1D) :** ajouter ≥2 PDE 1D au-delà de KS — p.ex. **Burgers** (chocs = états durs
  nets), **Korteweg–de Vries / Kuramoto-Sivashinsky variants**, **réaction-diffusion
  (Allen-Cahn / Fisher-KPP)**. Le bridge `al4pde` couvre déjà plusieurs PDE → coût surtout en
  config + validation.
- **P3.2 (2D) :** au moins une 2D — **Navier-Stokes 2D (vorticité)** ou **réaction-diffusion
  2D** (suites AL4PDE / PDEBench). Le générateur passe en conv2D ; la théorie est dimension-
  agnostique. C'est le saut qui crédibilise un A*.
- **Jalon G3 :** la hiérarchie *générateur > AL pool-based > i.i.d. uniforme > trajectoires
  uniformes* tient sur ≥3 PDE 1D + ≥1 PDE 2D, sur les métriques dures/rollout.

### Phase 4 — Baselines, rigueur, rédaction
- **P4.1** Baselines : (a) trajectoires uniformes, (b) états i.i.d. uniformes (décorrélation,
  I10), (c) **AL pool-based** (sélection dans pool fini — la nouveauté = synthèse hors-pool),
  (d) hard-mining par repondération d'importance (§3.2), (e) ablation `uniform_fraction`.
- **P4.2** Figure centrale : **métrique (rollout/val_hard) vs # appels solveur** (I10), 5 seeds
  pleins, tests appariés.
- **P4.3** Negative result documenté : conditionnement NRMSE (I5).
- **P4.4** Rédaction + reproductibilité (seeds, configs, scripts).

---

## 3. Claims → Figures (charpente du papier)

| Claim | Figure / Table | Sections théorie |
|---|---|---|
| C1. L'uniforme a une limite de **couverture** ; le gain est masqué par la métrique | val_hard/val_cov vs val_unif (Δ vs uniforme) + inégalité profil de loss | §7.1, §8.5 |
| C2. `uniform_fraction` = bouton DRO **borné** | Pareto bulk↔queue + courbe `R_p/R_q` vs `1/α` | §3.1, §7.2 |
| C3. Désaccord d'ensemble > loss comme signal | gain vs signal | §3.3 |
| C4. **Couverture du tube ⇒ rollout stable** (anti-exposure-bias) | horizon `K_τ`, `dist(û_k,M)` borné vs divergent | §9 |
| C5. **DAgger anticipé** (rollout-aware) > hard-mining one-step | `K_τ`, val_tube | §9.3–9.4 |
| C6. Data-efficiency : moins d'appels solveur à perf cible | métrique vs #solveur, 3 baselines | §2 (cadre) |
| C7. **Généralité** multi-PDE 1D+2D | hiérarchie de méthodes par PDE | toutes |

---

## 4. Risques & garde-fous

- **R1 — Réalisme du générateur (support hors `M`).** Si `q_gen` produit du hors-variété, on
  minimise un `R_∞` sur un `M` gonflé → gains illusoires. *Garde-fou :* `generator_metrics/`
  (TV, PSD, amplitude vs KS) + filtre solveur (états qui NaN sont rejetés).
- **R2 — Désalignement hard-mining ↔ tube visité.** Le hard one-step peut rater les états du
  rollout. *Garde-fou :* I9 (rollout-aware) + mesure `dist(û_k,M)`.
- **R3 — Coût des nouvelles métriques.** Eval post-hoc + tube coûtent. *Garde-fou :* post-hoc
  offline (déjà fait) découplé de l'entraînement.
- **R4 — `L_Φ > 1` (chaos KS).** Pas de borne uniforme du rollout, seulement ralentissement.
  *Garde-fou :* présenter le gain en **horizon stable**, pas en borne absolue ; choisir PDE/
  régimes où la garantie contractante existe par morceaux (Burgers post-choc, réac-diff).
- **R5 — Sur-promesse théorique.** Garder les énoncés *conditionnels* (H1/H2 explicites) ;
  l'honnêteté (negative results, conditions) est un atout en review A*.

---

## 5. Prochaines actions concrètes (ordre d'exécution)

1. **[Phase 1]** Lire le post-hoc complet (`/tmp/posthoc_full.txt`) → trancher G1. *(en cours)*
2. **[Phase 1]** Si G1 GO : figer le protocole de métriques durs comme métriques primaires.
3. **[Phase 2]** Créer configs : surrogate hidden64 + ensemble3 ; balayage `uniform_fraction` ;
   loss vs ensemble_var. Lancer 5 seeds.
4. **[Phase 2]** Implémenter le **tube de validation** + mesures `ε̄, K_τ, dist(û_k,M)`.
5. **[Phase 2]** Implémenter **rollout-aware hard mining** (I9).
6. **[Phase 4]** Baseline **états i.i.d. uniformes** + **AL pool-based** + figure efficacité-solveur.
7. **[Phase 3]** Étendre à Burgers / réac-diff 1D, puis une 2D.

> **Principe directeur anti-dispersion :** ne pas lancer la Phase 2+ tant que G1 (Phase 1)
> n'est pas tranché. La métrique dure est le prérequis : sans elle, tout gain reste « demi-
> teinte » par construction.
