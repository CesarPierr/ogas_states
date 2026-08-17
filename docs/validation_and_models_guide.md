# Guide Technique : Validation Pools, HPOs, et Architectures Génératives

Ce guide détaille les formulations mathématiques, les pipelines de génération de données, les hyperparamètres (HPOs) et l'architecture des modèles génératifs de l'infrastructure de boucle active sur le cluster Leonardo.

---

## 1. Génération des Pools de Validation

Tous les ensembles de validation sont dérivés de la **banque de validation uniforme** (`val_unif`), qui contient des transitions physiques simulées de l'équation de Kuramoto-Sivashinsky (KS) ou de Burgers :
* `states0` : états initiaux $s_t$ à l'étape $t$, de forme `[N, 1, L]`.
* `trajectories` : paires de transition $[s_t, s_{t+1}]$ de forme `[N, 2, 1, L]`.
* `params` : paramètres physiques de la PDE associés, de forme `[N, P]`.

---

### A. Ensemble Difficile par Variation Totale (`val_hard_tv_div`)
* **Fichier** : `val_hard_tv_div.npz`
* **Formulation Mathématique** : Le top 5% des transitions de la banque uniforme ayant la plus grande Variation Totale (TV) spatiale.
  Pour un état spatial périodique $u$ de dimension $L$ :
  $$\text{TV}(u) = \sum_{j=0}^{L-2} |u_{j+1} - u_j| + |u_0 - u_{L-1}|$$
* **Algorithme de Génération** :
  1. Calcul de $\text{TV}(s_t)$ pour chaque transition de la banque uniforme.
  2. Calcul du seuil au 95e percentile : $\text{TV}_{\text{seuil}} = q_{0.95}(\{\text{TV}_j\})$.
  3. Filtrage pour ne conserver que les candidats vérifiant $\text{TV}(s_t) \ge \text{TV}_{\text{seuil}}$.
  4. **Filtre de Diversité Temporelle** (`diversity_filter`) : Tri des candidats par TV décroissante. Sélection gloutonne d'au maximum 3 frames par trajectoire source, en imposant un écart temporel minimum de 10 pas de temps entre deux frames sélectionnées d'une même trajectoire.
  5. **Stratification des Paramètres** (`param_stratify`) : Répartition homogène des candidats sur une grille $4 \times 4$ de quantiles de paramètres $(\theta_0, \theta_1)$ pour éviter la concentration géographique des paramètres. Dans chaque cellule de paramètres, priorité est donnée aux états de plus haute TV.

---

### B. Ensemble Difficile par Faible Amplitude (`val_hard_lowamp_div`)
* **Fichier** : `val_hard_lowamp_div.npz`
* **Formulation Mathématique** : Le bottom 5% des transitions ayant la plus petite amplitude RMS (Root Mean Square) de leur état cible ($s_{t+1}$).
  $$\text{RMS}(s_{t+1}) = \sqrt{\frac{1}{L} \sum_{j=0}^{L-1} s_{t+1, j}^2}$$
* **Algorithme de Génération** :
  1. Calcul de $\text{RMS}(s_{t+1})$ pour chaque transition de la banque.
  2. Calcul du seuil au 5e percentile : $\text{RMS}_{\text{seuil}} = q_{0.05}(\{\text{RMS}_j\})$.
  3. Filtrage pour ne conserver que les candidats vérifiant $\text{RMS}(s_{t+1}) \le \text{RMS}_{\text{seuil}}$.
  4. **Filtre de Diversité Temporelle** : Tri par RMS croissante (plus l'amplitude est petite, plus la priorité est haute). Sélection d'au maximum 3 frames par trajectoire source, avec un écart temporel minimum de 10 pas de temps.
  5. **Stratification des Paramètres** : Répartition des candidats sur la grille $4 \times 4$ de quantiles de paramètres, en favorisant les candidats de plus faible amplitude au sein de chaque cellule.

---

### C. Ensembles de Tubes Multi-Bandes (Dérive Hors-Attracteur)
* **Fichiers** : `val_tube_{low,mid,high}k_rho{0p1,0p25,0p5}.npz`
* **Objectif** : Mesurer la stabilité locale et globale en dehors de l'attracteur physique en évaluant la capacité de rappel du surrogate vers le manifold.
* **Formulation Mathématique** :
  On sélectionne des états d'ancrage sains $u_{\text{base}}$ sur l'attracteur (seconde moitié temporelle des trajectoires de la banque uniforme).
  On génère un bruit spatial périodique $\eta(x)$ dans le domaine de Fourier :
  $$\eta(x) = \sum_{k=k_{\text{min}}}^{k_{\text{max}}} \frac{1}{k} \left( A_k \cos(kx) + B_k \sin(kx) \right), \quad A_k, B_k \sim \mathcal{N}(0, 1)$$
  Ce bruit est normalisé pour avoir une amplitude RMS exactement égale à 1 :
  $$\bar{\eta}(x) = \frac{\eta(x)}{\text{RMS}(\eta)}$$
  L'état perturbé hors-attracteur $u_{\text{pert}}$ est construit par :
  $$u_{\text{pert}} = u_{\text{base}} + \rho \cdot \text{RMS}(u_{\text{base}}) \cdot \bar{\eta}$$
  où $\rho \in \{0.1, 0.25, 0.5\}$ est le facteur d'échelle de la perturbation relative.
* **Calcul du Target** : L'état perturbé $u_{\text{pert}}$ est résolu d'un pas de temps exact par le solveur JAX pour donner $u_{\text{pert}, t+1}$.
* **Bandes de Nombres d'Onde** :
  * **Low-k** (`lowk`) : $k \in [1, 12]$ (grandes perturbations de structure)
  * **Mid-k** (`midk`) : $k \in [13, 60]$ (perturbations à moyenne échelle)
  * **High-k** (`highk`) : $k \in [61, 200]$ (bruit spatial à très haute fréquence)

---

## 2. Hyperparamètres Clés de l'Entraînement (HPO)

| Paramètre | Valeur | Description |
|---|---|---|
| **Rounds** | 10 | Nombre de boucles d'apprentissage actif |
| **Trajectoires par Round** | 250 | Nombre de simulations générées à chaque round |
| **Pas de temps par trajectoire** | 70 | Nombre d'étapes simulées par trajectoire |
| **Transitions par Round** | 17 500 | Nombre total d'exemples de transition par round |
| **Fraction Uniforme ($\alpha$)** | 0.50 | 50% de données d'attracteur uniforme, 50% de données générées |
| **Époques du Surrogate / Round** | 15 | Époques d'entraînement du modèle surrogate |
| **Taille de Batch du Surrogate** | 128 | Batch size du surrogate |
| **Learning Rate du Surrogate** | 0.001 | Taux d'apprentissage (AdamW) du surrogate |
| **Époques du Générateur / Round** | 10 | Époques d'entraînement du modèle génératif |
| **Taille de Batch du Générateur** | 128 | Batch size du générateur |
| **Learning Rate du Générateur** | 0.0002 | Taux d'apprentissage (AdamW) du générateur |
| **Nombre de Quantiles de Perte** | 20 | Division de l'intensité de perte (difficulté) en 20 bins |
| **SDEdit $t_0$** | 0.60 | Ratio de re-bruitage pour le mode échantillonnage "edit" |
| **Candidate Oversampling** | 4 | Facteur de suréchantillonnage de candidats générés ($4\times$) |
| **TV Gate Threshold** | 3.0 | Élimination des états générés avec TV $> 3 \times \text{p99}_{\text{uniform\_TV}}$ |

---

## 3. Architecture des Modèles Génératifs

Les deux classes de générateurs (`DDPM1D` et `FlowMatching1D`) partagent le même réseau de convolution 1D avec conditions aux limites périodiques (circular padding).

### A. MLP de Conditionnement (`self.cond`)
Avant d'entrer dans le réseau de convolutions, les variables de conditionnement sont fusionnées en un unique vecteur de conditionnement `cond_vec` de taille `hidden = 64` via un MLP à 2 couches :
1. **Embedding Temporel** : L'étape de diffusion $t$ est projetée sur un embedding sinusoïdal de taille `hidden = 64`.
2. **Embedding de Difficulté** : Le bin de quantile de perte (0 à 19) est projeté sur un embedding appris de taille `quant_embed_dim = 8`.
3. **Embedding des Paramètres PDE** : Les paramètres physiques $\theta$ (taille $P$) sont concaténés directement.
4. **Embedding d'Amplitude Target** (Vague 4 / Option B) : L'amplitude de la cible (normalisée par `state_std`) est passée comme valeur scalaire 1D.

$$\text{Entrée MLP} = [\text{Sinusoidal Time} \oplus \text{Loss Quantile Embedding} \oplus \text{PDE Params} \oplus \text{Target Amplitude}]$$

Ces tenseurs passent par :
$$\text{Linear}(D_{\text{in}}, 64) \to \text{SiLU}() \to \text{Linear}(64, 64) \to \text{cond\_vec}$$

---

### B. Réseau Convolutionnel (`ResidualDDPMNet1D`)
Le réseau principal convertit un état bruité $x_t \in \mathbb{R}^{1 \times 1 \times L}$ en champ de bruit ou de vitesse prédit :
1. **Projection d'Entrée** : `Conv1d` avec padding circulaire, projetant le canal d'état et le conditionnement spatialisé de $1 + 64$ canaux vers $64$ canaux.
2. **Blocs Résiduels** : $N = 2$ blocs de convolution résiduels (`ResidualConvBlock1D`).
   * Les convolutions utilisent des dilatations espacées ($1, 2, 4 \dots$) pour capter les relations spatiales à longue distance sans réduction de dimension (pas de pooling).
   * **Conditionnement FiLM** : Si `cond_mode = "film"`, le vecteur `cond_vec` est projeté linéairement pour prédire un facteur d'échelle ($\gamma$) et un décalage ($\beta$) pour chaque canal intermédiaire de la convolution, appliquant l'opération :
     $$h_{\text{new}} = h \cdot (1 + \gamma) + \beta$$
3. **Projection de Sortie** : `Conv1d` finale (kernel size 7) pour projeter les 64 canaux de sortie vers 1 canal physique d'état.

---

## 4. Mécanismes de Boucle Active : Choix des Cibles, SDEdit et Sélection

Lors du déroulement des rounds actifs ($Round \ge 1$), la génération et la sélection des nouveaux états d'entraînement suivent un processus en 4 étapes clés :

### A. Choix des Quantiles de Difficulté Cibles
Le générateur modélisant $p(\text{state} \mid \text{loss\_quantile})$, on doit lui fournir une cible de difficulté (quantile bin) lors de la génération.
* **Fonction** : `sample_quantile_labels` (dans [train.py](file:///poolbased_surrogate/train.py#L30))
* **Stratégie** : `top_half` (par défaut). Elle tire des bins uniformément au hasard parmi la moitié supérieure des quantiles de difficulté de la banque :
  $$\text{bin} \sim \mathcal{U}([q/2, q-1])$$
  Pour 20 quantiles, cela cible uniquement des indices de bins de $10$ à $19$ (les 50% d'états les plus difficiles).

### B. Mode Édition / Prior de Manifold (SDEdit)
Générer à partir de bruit pur (mode `scratch`) produit des états trop décorrélés de la physique réelle. Le mode `edit` contraint la génération autour d'ancres de l'attracteur.
* **Fonction** : `sample` (dans [ddpm.py](file:///poolbased_surrogate/models/ddpm.py#L652))
* **Fonctionnement** :
  1. On tire un état physique sain $u_{\text{base}}$ sur l'attracteur (nos ancres).
  2. On applique un re-bruitage partiel au niveau $t_0 = 0.60$ :
     $$x(t_0) = (1 - t_0) \cdot \epsilon + t_0 \cdot u_{\text{base}}$$
     où $\epsilon$ est le bruit de base (prior de spectre).
  3. L'intégration de Flow Matching démarre de $t_0 = 0.60$ jusqu'à $t = 1.0$ (au lieu de démarrer de $t=0.0$).
  4. L'état généré conserve ainsi la macro-structure de l'ancre $u_{\text{base}}$ tout en adaptant les hautes fréquences selon les conditions demandées (perte, amplitude).

### C. Suréchantillonnage et Barrière de Réalisme (TV Gate)
Pour un batch de taille $N$ nécessaire, le système génère un suréchantillonnage de candidats :
$$\text{want} = N \times \text{candidate\_factor} \quad (\text{avec } \text{candidate\_factor} = 4)$$
Pour éliminer les candidats trop rugueux (bruit blanc hors-manifold), on applique une barrière adaptative de variation totale (TV Gate) :
1. Calcul du seuil dynamique :
   $$\text{TV}_{\text{threshold}} = \text{realism\_tv\_gate} \times q_{0.99}(\{\text{TV}(u_{\text{anchors}})\}) \quad (\text{avec } \text{realism\_tv\_gate} = 3.0)$$
2. Tout candidat dont la Variation Totale dépasse $\text{TV}_{\text{threshold}}$ est éliminé.

### D. Sélection par Désaccord d'Ensemble (Steering Réalisé)
Parmi les candidats survivant au TV Gate, on effectue une sélection active :
1. On passe les candidats dans l'ensemble de surrogates U-Net pour calculer leur variance de prédiction spatialisée (le désaccord d'ensemble) :
   $$\text{dis} = \text{Var}_{\text{ensemble}}(\tilde{\mathcal{M}}(s_t; \theta))$$
2. On trie les candidats par désaccord décroissant et on ne garde que les $N$ premiers.
3. Ce mécanisme permet de cibler les états où le surrogate est le plus incertain sans coût de calcul supplémentaire pour le solveur.

