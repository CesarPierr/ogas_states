
## 1. Tableau Récapitulatif des Performances (Round 10)

Les performances ont été moyennées sur **5 graines aléatoires** au round 10 (budget de solveur équilibré et apparié). Les pourcentages indiquent le changement relatif par rapport au contrôle (Uniform Baseline).

| Variante d'Échantillonnage (Arm) | Bulk (mean RMSE) | TV-Hard (p99 RMSE) | Tube ($\rho = 0.1$) | Tube ($\rho = 0.5$) | Rollout @10 pas (NRMSE) | Rollout @50 pas (NRMSE) | Rollout @100 pas (NRMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform Baseline** (contrôle, $\alpha=1$) | 0.0035 | 0.0543 | 0.0130 | 0.0415 | 5.452 | 6.625 | 7.157 |
| **$\alpha = 0.0$** (tophalf) | 0.0107 (+209%) | 0.0593 (+9.2%) | 0.0109 (-15.9%) | 0.0149 (-64.1%) | 2.899 (-46.8%) | 6.458 (-2.5%) | 8.480 (+18.5%) |
| **$\alpha = 0.25$** (tophalf) | 0.0045 (+28.5%) | 0.0219 (-59.6%) | 0.0047 (-63.5%) | 0.0093 (-77.7%) | 2.972 (-45.5%) | 6.028 (-9.0%) | 6.976 (-2.5%) |
| **$\alpha = 0.5$** (tophalf) | 0.0052 (+51.0%) | 0.0307 (-43.4%) | 0.0059 (-54.2%) | 0.0110 (-73.4%) | 2.257 (-58.6%) | 5.268 (-20.5%) | 6.754 (-5.6%) |
| **$\alpha = 0.75$** (tophalf) | 0.0037 (+6.5%) | 0.0263 (-51.5%) | 0.0049 (-62.0%) | 0.0095 (-77.1%) | 3.009 (-44.8%) | 18.017 (+172%) | 73.158 (+922%) |
| **$\alpha = 0.5$** (**ensvar**) | **0.0034** (-1.2%) | **0.0182** (-66.4%) | **0.0042** (-67.9%) | **0.0088** (-78.8%) | **2.153** (-60.5%) | **4.063** (-38.7%) | **5.199** (-27.4%) |

---

## 2. Détails des Ensembles de Validation Benchmark (Set A & Set B)

Dans la Vague 2, trois familles de jeux de validation ont été construites pour auditer les forces et faiblesses du surrogate :

### A. Le Pool Uniforme (`val_unif`)
* **Objectif** : Mesurer l'erreur sur l'attracteur physique normal (performance moyenne ou *bulk*).
* **Génération** : Simulation standard par le solveur exact à partir de conditions initiales de Fourier aléatoires. Les paramètres physiques sont distribués uniformément.

### B. Les Pools Difficiles ("Hard sets")
Ces pools isolent les dynamiques d'états extrêmes rencontrées sur l'attracteur. Dans la Vague 2, l'accent a été mis sur :

**TV-Hard (`val_hard_tv`)** :
    *   **Définition** : États du top 5% des variations totales de la banque.
    *   **Formulation** :
        $$\text{TV}(u) = \sum_{j} |u_{j+1} - u_j|$$
    *   **Physique** : Isole les chocs physiques et les forts gradients spatiaux.
<!-- 2.  **Low-Amplitude (`val_hard_lowamp`)** :
    *   **Définition** : États du bottom 5% en énergie/amplitude RMS.
    *   **Formulation** :
        $$\text{RMS}(u) = \sqrt{\frac{1}{L} \sum_{j} u_j^2}$$
    *   **Physique** : Les zones calmes de basse énergie (où tous les surrogates d'apprentissage actif sans correction d'amplitude échouent en raison du biais de perte absolue). -->

### C. Les Pools de Tube (`val_tube_rho*`)
Ces ensembles évaluent la robustesse face à la dérive numérique (manifold drift) lors de simulations à long terme.
* **Génération** :
  Des états sains de l'attracteur $u_{\text{base}}$ sont perturbés par un bruit de Fourier de spectre $1/k$ limité aux grandes structures ($k \le 12$), normalisé à un RMS de 1 ($\bar{\eta}$) :
  $$u_{\text{pert}} = u_{\text{base}} + \rho \cdot \text{RMS}(u_{\text{base}}) \cdot \bar{\eta}$$
* **Échelles $\rho$** :
  *   $\rho = 0.1$ : Test de stabilité locale proche de l'attracteur.
  *   $\rho = 0.5$ : Test robuste de rappel vers le collecteur (perturbations très fortes).

---

## 3. Description des Variantes Comparées (Benchmark)

1.  **Uniform Baseline ($\alpha = 1.0$)** :
    *   Le contrôle classique. Pas de générateur. Le pool d'entraînement à chaque round est reconstruit uniquement avec des trajectoires uniformes fraîches résolues par le solveur.
2.  **Mélange Uniforme $\alpha \in \{0.0, 0.25, 0.5, 0.75\}$ (`tophalf`)** :
    *   Le pool d'entraînement contient une fraction $\alpha$ de trajectoires uniformes et une fraction $1 - \alpha$ d'états générés activement.
    *   Les états générés ciblent la moitié supérieure des quantiles de perte absolue (`tophalf`, c'est-à-dire les bins $10 \dots 19$ sur 20).
    *   *Remarque* : $\alpha = 0.0$ est l'apprentissage actif pur (sans ancrage uniforme).
3. **Désaccord d'Ensemble $\alpha = 0.5$ (`ensvar`)** :
    *   L'ablation cruciale. Au lieu d'entraîner et de conditionner le générateur sur la perte un pas du surrogate (`conditional_loss`), le système utilise la variance spatiale des prédictions de l'ensemble de surrogates U-Net (disagreement) comme signal de difficulté.

---

## 4. La Barrière de Réalisme : Le TV Gate

Le **TV Gate** (ou barrière de variation totale) est un filtre de réalisme physique appliqué immédiatement après la génération des candidats par le DDPM/Flow Matching, avant l'étape de sélection par désaccord d'ensemble.

### A. Objectif Physique
Les modèles génératifs de type diffusion peuvent parfois produire des états aberrants contenant du bruit blanc ou des oscillations à très haute fréquence (artéfacts géométriques). 
Si l'on envoie ces états bruités au solveur exact (JAX), cela peut provoquer des divergences numériques. De plus, ces états ne représentent pas la physique réelle de la PDE, qui est naturellement lissée par les termes de dissipation (par exemple, le terme $-u_{xxxx}$ dans Kuramoto-Sivashinsky).

### B. Formulation Mathématique

Pour un état physique généré $u \in \mathbb{R}^L$ de dimension $L$ sous conditions aux limites périodiques, on calcule sa **Variation Totale (TV)** :
$$\text{TV}(u) = \sum_{j=0}^{L-2} |u_{j+1} - u_j| + |u_0 - u_{L-1}|$$

La Variation Totale mesure la somme des sauts de valeur d'un point de grille à l'autre. Un état physique fluide aura une TV modérée, tandis qu'un état bruité avec des dents de scie présentera une TV extrêmement élevée.

### C. Seuil Dynamique et Filtrage

Plutôt que de fixer un seuil arbitraire, le TV Gate calcule un seuil **dynamique** basé sur les vrais états de l'attracteur physique (les ancres $u_{\text{anchors}}$ issues des trajectoires uniformes) :

1. On extrait la TV des états réels de l'attracteur et on calcule le 99e percentile : $q_{0.99}(\{\text{TV}(u_{\text{anchors}})\})$.
2. On applique un multiplicateur de sécurité défini dans la configuration (`realism_tv_gate = 3.0`) :
   $$\text{TV}_{\text{threshold}} = 3.0 \times q_{0.99}(\{\text{TV}(u_{\text{anchors}})\})$$
3. Pour chaque candidat généré $u_{\text{gen}}$, si $\text{TV}(u_{\text{gen}}) > \text{TV}_{\text{threshold}}$, le candidat est **rejeté**.

*Mécanisme de secours (fail-safe)* : Si le filtre se montre trop strict et rejette trop de candidats au point de ne plus avoir assez d'états pour le batch requis, le TV Gate se relâche automatiquement et accepte tous les états qui ne contiennent pas de valeurs non finies (`NaN` ou `Inf`), afin de ne pas bloquer la boucle d'apprentissage.

---

## 5. Ablation de la Sélection Active : Ce que chaque baseline isole

### A. Tableau des baselines et ce qu'elles décomposent

Le pipeline complet de notre méthode (`gen_v3_edit`) fait trois choses :
1. **Génération apprise** : le DDPM/Flow Matching propose des états ciblant les zones difficiles.
2. **Filtre de réalisme** : le TV Gate élimine les artéfacts non physiques.
3. **Sélection active** : le désaccord d'ensemble choisit les $N$ états les plus incertains parmi les $4N$ candidats survivants.

Pour décomposer la contribution de chaque composant, les baselines suivantes ont été testées en Vague 3 :

| Baseline | Source des états candidats | Sélection par désaccord ? | Ce qu'elle isole |
| :--- | :--- | :---: | :--- |
| `uniform_baseline` | Trajectoires uniformes complètes | Non | Contrôle (aucun active learning) |
| `random_tube` | Perturbations hors-attracteur aléatoires d'ancres uniformes | Non | Effet de l'injection de données hors-manifold brute |
| `tube_select` | Perturbations hors-attracteur aléatoires d'ancres uniformes | **Oui** | Valeur de la sélection par désaccord quand les candidats sont hors-attracteur |
| `mined_ic` | Conditions initiales de Fourier aléatoires (`sample_ic_uniform`) | **Oui** | Valeur de la sélection par désaccord quand les candidats sont des ICs brutes (pas sur l'attracteur non plus) |
| `gen_pure_edit` | DDPM en mode SDEdit (ancre + re-bruitage) | Non (TV Gate seul) | Qualité du générateur seul (sans sélection active) |
| `gen_v3_edit` | DDPM en mode SDEdit | **Oui** (TV Gate + désaccord) | Pipeline complet |

### B. Résultats comparatifs (Vague 3, NRMSE mean vs. `random_tube`)

| Baseline | Bulk (val_unif) | Tube midk $\rho=0.5$ | Tube lowk $\rho=0.5$ | Tube highk $\rho=0.5$ |
| :--- | :---: | :---: | :---: | :---: |
| `tube_select` | +41,6% | -11,2% (n.s.) | +33,1% | -5,2% (n.s.) |
| `mined_ic` | +111,3% | +17,7% | +107,3% | +9,9% |
| `gen_pure_edit` | -5,2% (n.s.) | -59,6% | -0,4% (n.s.) | -55,2% |
| **`gen_v3_edit`** | **-27,5%** | **-82,6%** | **-27,7%** | **-45,4%** |

### C. Ce que ces résultats nous apprennent

**1. `tube_select` ≠ sélection sur des trajectoires uniformes.**
L'agent précédent a confondu `tube_select` avec une sélection dans les trajectoires uniformes. En réalité, `tube_select` génère des états **perturbés hors-attracteur** (par la même perturbation de Fourier basse fréquence que `random_tube`), puis sélectionne les plus incertains par désaccord d'ensemble. Les candidats ne sont pas des états issus de trajectoires uniformes sur l'attracteur ; ce sont des états volontairement éjectés du collecteur.

**2. La baseline manquante que vous identifiez.**
La baseline que vous suggérez — « prendre des états issus de trajectoires uniformes **sur l'attracteur**, calculer le désaccord d'ensemble dessus, et ne garder que les 50% les plus incertains » — n'a pas été testée. C'est pourtant une ablation essentielle car elle isolerait proprement la valeur ajoutée du modèle génératif par rapport à un simple **tri actif des données d'attracteur disponibles**. Elle serait plus compétitive que `tube_select` ou `mined_ic` car :
- Les candidats sont *physiquement corrects* (ils sont sur l'attracteur, pas des ICs aléatoires ou des perturbations hors-manifold).
- La sélection par désaccord devrait correctement identifier les sous-régions de l'attracteur les moins bien apprises.
- C'est exactement le régime « pool-based active learning classique sur données uniformes ».

**3. Pourquoi `mined_ic` échoue malgré la sélection.**
`mined_ic` génère des conditions initiales de Fourier aléatoires (`sample_ic_uniform`), qui sont des superpositions de sinus/cosinus à coefficients aléatoires. Ces états ne sont **pas sur l'attracteur** (ils n'ont pas été intégrés par le solveur). L'ensemble de surrogates a naturellement un fort désaccord sur ces données hors-distribution, mais les entraîner dessus ne prépare pas le modèle aux vraies dynamiques physiques. Le signal de désaccord est « pollué » par le caractère non physique des candidats.

**4. Pourquoi le générateur est malgré tout nécessaire (au-delà de la sélection).**
Le fait que `gen_pure_edit` (générateur sans sélection active) obtienne des résultats comparables ou supérieurs à `tube_select` (pas de générateur mais avec sélection) montre que le prior physique appris par le DDPM est une composante à part entière. Le DDPM ne se contente pas de "proposer des états où le surrogate a du mal" : il apprend la géométrie du collecteur et sait y placer des états physiquement cohérents dans les zones sous-représentées. La sélection active amplifie ce signal, mais ne le crée pas.

---

## 6. Baseline manquante à tester : `uniform_select`

### Proposition
Implémenter une nouvelle stratégie `uniform_select` qui :
1. Tire $4N$ états depuis les trajectoires uniformes existantes (ancres saines de l'attracteur).
2. Calcule le désaccord d'ensemble sur ces $4N$ états.
3. Garde les $N$ états ayant le plus fort désaccord pour constituer la moitié non-uniforme du pool.

### Intérêt scientifique
Cette baseline répondrait à la question : **le modèle génératif apporte-t-il quelque chose au-delà d'un simple tri actif des données d'attracteur ?** Si `uniform_select` rivalise avec `gen_v3_edit`, cela signifierait que le DDPM est redondant avec la sélection par désaccord. Si `gen_v3_edit` surpasse nettement `uniform_select`, cela prouverait que le générateur explore des régions de l'espace d'état que les trajectoires uniformes ne couvrent tout simplement jamais, même en les filtrant intelligemment.

<!-- ---


## 4. Focus Mathématique : Le Désaccord d'Ensemble (ensvar)

Le désaccord d'ensemble (`ensvar`) s'est révélé être la variante la plus robuste de la Vague 2. Ce mécanisme repose sur la modélisation de l'**incertitude épistémique** plutôt que sur l'erreur instantanée.

### A. Formulation Mathématique

Le surrogate est constitué d'un ensemble de $M$ réseaux de neurones U-Net indépendant (par exemple $M = 3$) :
$$\mathcal{M} = \{ \tilde{\mathcal{M}}_1, \tilde{\mathcal{M}}_2, \dots, \tilde{\mathcal{M}}_M \}$$

Pour un état physique d'entrée $s_t \in \mathbb{R}^L$ et un paramètre de PDE $\theta \in \mathbb{R}^P$, chaque membre de l'ensemble $i$ prédit un état cible :
$$\hat{s}^{(i)} = \tilde{\mathcal{M}}_i(s_t; \theta) \quad \in \mathbb{R}^L$$

La moyenne des prédictions de l'ensemble est définie par :
$$\mu(s_t; \theta) = \frac{1}{M} \sum_{i=1}^M \hat{s}^{(i)}$$

Le **désaccord d'ensemble** (ou variance d'ensemble spatiale) correspond à la variance moyenne des prédictions des membres sur toute la grille spatiale de taille $L$ :
$$\text{Var}_{\text{ensemble}}(s_t; \theta) = \frac{1}{L} \sum_{j=1}^L \left( \frac{1}{M} \sum_{i=1}^M \left( \hat{s}^{(i)}_j - \mu_j(s_t; \theta) \right)^2 \right)$$

### B. Pourquoi ensvar surclasse-t-il le conditionnement par perte (loss) ?

1. **Pas de besoin de solveur (Self-Supervised / Active Exploration)** :
   * Pour évaluer la perte absolue d'un candidat généré, le système a besoin de l'état cible exact $s'_{t+1}$, ce qui exige un appel coûteux au solveur PDE.
   * La variance d'ensemble, elle, ne dépend *que* des prédictions des U-Nets. Elle peut être calculée instantanément sur des milliers d'états candidats.
2. **Ciblage de l'incertitude pure (Query-by-Committee)** :
   * La perte un pas (`loss`) pousse le générateur à reproduire des états complexes (zones turbulentes), même si le surrogate sait déjà bien les prédire.
   * La variance d'ensemble (`ensvar`) cible uniquement les états où les réseaux de l'ensemble *ne sont pas d'accord*. Cela dirige le générateur vers les régions inexplorées de l'espace d'état (exploration active). -->

<!-- ---

## 5. Pourquoi n'a-t-on pas les performances du dataset Low-Amplitude dans le tableau principal ?

Il y a deux raisons fondamentales — l'une historique, l'autre physique — pour lesquelles les performances sur le dataset `val_hard_lowamp` n'apparaissent pas dans le tableau principal des résultats de la Vague 2 :

### A. Raison Historique (Chronologie de la Vague 2)
* Le pool de validation initial de la Vague 2 ne comportait pas de dataset spécifique aux faibles amplitudes. Le benchmark se concentrait uniquement sur le pool uniforme (`val_unif`), les chocs à hauts gradients (`val_hard_tv`) et la stabilité face à la dérive numérique (les ensembles `val_tube`).
* Le dataset `val_hard_lowamp` a été créé **a posteriori (en fin de sweep de la Vague 2)** comme outil de diagnostic, lorsque l'équipe a cherché à comprendre pourquoi les rollouts à long terme de toutes les variantes d'apprentissage actif commençaient à accumuler d'importantes dérives dans les phases calmes ou à basse énergie.

### B. Raison Physique (La Régression de Basse Amplitude)
* Les variantes de la Vague 2 s'appuyaient uniquement sur des métriques d'erreur **absolues** (RMSE un pas pour `loss`, ou variance de prédiction absolue pour `ensvar`).
* Dans une zone de faible énergie ou d'ondes calmes, l'état physique $u$ a une valeur numérique proche de zéro. Par conséquent, l'erreur absolue y est structurellement minuscule, même si le modèle commet une erreur relative énorme (par exemple, prédire du bruit ou des oscillations fantômes représentant +200% d'erreur relative).
* Le générateur guidé par la perte absolue a donc classé ces états calmes comme "très faciles" (perte absolue $\approx 0$) et n'a jamais proposé de transitions à basse amplitude pour réentraîner le modèle.
* **Le résultat de l'audit** : Le surrogate a subi une déprivation de données à basse amplitude, provoquant une régression catastrophique (erreur relative en hausse de **+18% à +189%** par rapport à la référence uniforme). 

### C. Pourquoi ces chiffres n'étaient pas dans le tableau de publication ?
* Le tableau principal de la publication scientifique a été conçu pour mettre en valeur les bénéfices nets et robustes de l'apprentissage actif (les réductions massives d'erreurs sur TV-Hard et sur les Tubes).
* La régression sur les faibles amplitudes, étant un **résultat négatif** et un problème ouvert (un bug d'échantillonnage), a été exclue du tableau de performance générale pour être traitée à part dans les sections de discussion sur les perspectives d'amélioration ("adverse findings").
* C'est précisément pour cela que la **Vague 4** a été lancée : introduire des correctifs d'amplitude (normalisation par NRMSE et conditionnement RMS) pour redonner au générateur la visibilité sur ces zones calmes et corriger cette régression.

 -->
