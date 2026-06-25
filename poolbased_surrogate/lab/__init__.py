"""Loss-generator lab : banc d'essai OFFLINE du conditionnement sur problème FIGÉ.

Sous-paquet self-contained, indépendant de la boucle d'entraînement ([run.py]) : il construit
un dataset (état, difficulté) à partir d'un run terminé + surrogate GELÉ, puis entraîne / sweep
des générateurs loss-conditionnés dessus. C'est ici qu'on a établi que le conditionnement
fonctionne sur le problème statique mais s'effondre en production (cible mobile = moving target).

Modules :
  common   — chargement config + helpers partagés
  dataset  — construit le dataset figé (état, quantile de loss) depuis un run + surrogate gelé
  generator— construction / sampling du générateur
  training — entraîne UN générateur loss-conditionné
  scoring  — score les échantillons vs stats du vrai KS
  sweep    — sweep d'architectures / hyperparamètres
  lab      — orchestrateur + CLIs (build_dataset_cli / train_cli / sweep_cli)

Points d'entrée : scripts/{build,train,sweep}_loss_generator.py et scripts/sweep_conditioning_hpo.py
"""
from .lab import build_dataset_cli, sweep_cli, train_cli
from .sweep import run_sweep

__all__ = ["build_dataset_cli", "sweep_cli", "train_cli", "run_sweep"]
