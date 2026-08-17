# Mini-MOOC — découverte du projet OGAS states

Ce parcours sert à rentrer vite dans le projet sans devoir lire tout le dépôt en vrac. Il est conçu pour être suivi dans l’ordre, avec les notebooks comme support pratique.

## Objectif du projet

Le projet teste une idée précise : entraîner un surrogate neuronal de PDE 1D en remplaçant une partie des trajectoires uniformes par des états générés, ciblés vers les zones difficiles, puis labellisés par le solveur exact.

Une donnée d’entraînement est toujours une transition :

```text
(state_t, pde_parameter) -> state_{t+1}
```

Invariant important :

```text
n_samples = pool.trajectory_steps * pool.n_trajectories
```

## Parcours recommandé

1. `notebooks/mooc_00_project_map.ipynb`
   - comprendre la structure du dépôt ;
   - lire les configs, les modules centraux et les artefacts de run ;
   - savoir où chercher quand on veut modifier une expérience.

2. `notebooks/mooc_01_data_pde_pool.ipynb`
   - inspecter `validation.npz` et `pool_round_*.npz` ;
   - visualiser les états, les transitions, les pertes et les sources ;
   - comprendre uniform vs generated.

3. `notebooks/mooc_02_results_investigations.ipynb`
   - lire `history.json` ;
   - comparer les métriques `val/*`, `rollout/*`, `pool/*`, `ddpm/*` ;
   - relier les résultats locaux aux conclusions du handoff.

4. `notebooks/mooc_03_research_next_steps.ipynb`
   - transformer l’état actuel en plan de recherche ;
   - choisir les prochaines campagnes ;
   - générer des commandes de validation, post-hoc et analyse.

## Carte mentale rapide

- `poolbased_surrogate/run.py` : boucle expérimentale complète.
- `poolbased_surrogate/pool.py` : construction des pools uniformes, générés, tube et mined IC.
- `poolbased_surrogate/train.py` : entraînement surrogate + générateur, difficulté et bins.
- `poolbased_surrogate/eval.py` : validation one-step et rollout.
- `poolbased_surrogate/models/surrogate.py` : ensemble de surrogates Conv1D.
- `poolbased_surrogate/models/ddpm.py` : DDPM et flow matching 1D.
- `scripts/build_*validation.py` : validation dure, tube et diverse suite.
- `scripts/eval_*posthoc.py` : analyses post-hoc sur checkpoints.
- `configs/*` : protocoles expérimentaux.
- `HANDOFF.md` : résumé de recherche, résultats établis, risques et prochaines étapes.

## Résultats actuels à retenir

D’après `HANDOFF.md` :

- Phase 2 KS propre : les variantes ciblées améliorent fortement les validations tube et TV-hard.
- Le bulk uniforme n’est pas systématiquement meilleur ; ce n’est pas la métrique principale.
- Le set lowamp-hard est un résultat négatif important : toutes les variantes empirent.
- Les anciens résultats Burgers sont invalides côté validation : la banque contient des artefacts solveur et doit être régénérée avec un stability gate.
- La v3 corrige plusieurs défauts : difficulté pré-entraînement, stats EMA, réduction du self-feeding, bruit spectral, gate TV, oversample-then-reject et SDEdit optionnel.

## Manière raisonnable de continuer

1. Reproduire localement un `smoke` pour vérifier l’environnement.
2. Inspecter les runs existants avec les notebooks.
3. Regénérer les validations dures/diverses quand le protocole change.
4. Comparer d’abord à budget solveur apparié.
5. Exiger que le générateur batte `random_tube` et `mined_ic`, sinon la génération apprise n’est pas justifiée.
6. Garder les résultats négatifs dans l’analyse : lowamp-hard, bulk non-neutre, corrélation de conditionnement faible.

## Commandes utiles

```bash
.venv/bin/python -m poolbased_surrogate.run configs/smoke.yaml --fresh
```

```bash
.venv/bin/python scripts/build_diverse_validation.py \
  --bank <validation_uniforme.npz> \
  --config <config.yaml> \
  --output-dir <validation_diverse_dir>
```

```bash
.venv/bin/python scripts/eval_hard_posthoc.py \
  --runs <run_dir_1> <run_dir_2> \
  --uniform <validation_uniforme.npz> \
  --hard <validation_diverse_dir> \
  --baseline uniform_baseline
```

```bash
.venv/bin/python scripts/eval_rollout_posthoc.py \
  --runs <run_dir_1> <run_dir_2> \
  --uniform <validation_uniforme.npz> \
  --steps 100 \
  --baseline uniform_baseline
```
