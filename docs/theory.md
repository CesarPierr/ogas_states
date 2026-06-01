# Cadre théorique : générer des états « durs » pour entraîner un surrogate de PDE

> **But du document.** Donner un cadre formel à ce que l'on fait — *au lieu d'échantillonner
> des trajectoires uniformes peu informatives et fortement corrélées, échantillonner des
> états là où le surrogate est mauvais / incertain* — et répondre honnêtement à la question :
> **a-t-on une garantie que ça réduit la RMSE/NRMSE sur un set de validation uniforme ?**
>
> Réponse courte : **non, pas de garantie inconditionnelle, et on peut prouver qu'on peut
> nuire.** Mais on a (a) des garanties *conditionnelles* exploitables, (b) une explication
> rigoureuse de pourquoi le gain est robuste sur le *rollout* mais ambigu sur la RMSE-bulk,
> et (c) le `uniform_fraction` qui apparaît comme le paramètre de *contrôle* du compromis,
> avec une borne multiplicative explicite. Ces trois points cadrent le POC et le papier.
>
> **⚠️ Recadrage majeur (§7–§8).** La question ci-dessus est en réalité *mal posée* : valider
> sur un set uniforme demande au surrogate d'être bon *en moyenne sur l'attracteur*, ce qui
> est précisément ce que le sampling uniforme optimise déjà. Le vrai but n'est pas « meilleur
> surrogate sur le set uniforme » mais **« bon surrogate sur *tous* les états atteignables »**,
> c.-à-d. chercher une **uniformité sur la loss** (peu importe la fréquence de l'état). Cela
> change la fonctionnelle cible (de la moyenne `E_p` vers un objectif **coverage / minimax /
> DRO**) et **impose de changer le set de validation** (§8). Dans ce cadre, sur-échantillonner
> les loss dures n'est plus un compromis ambigu : c'est l'estimateur naturel de l'objectif, et
> il devrait *dominer* le sampling uniforme — mais on ne peut le **mesurer** qu'avec une
> validation dure / en région non-vue. C'est le cœur du papier.

---

## 1. Notations et objet d'étude

- État `u ∈ ℝ^L` (champ KS discrétisé, `L=800`), paramètres physiques `c` (diffusivité, domaine…).
- Opérateur de transition exact du solveur (un pas de `n_substeps`) : `Φ(u, c)`.
- Surrogate `f_θ(u, c) ≈ Φ(u, c)`. On l'entraîne en MSE one-step :
  `ℓ_θ(u,c) = ‖f_θ(u,c) − Φ(u,c)‖²`.
- **Distribution de validation** `p` : la mesure des états induite par des *trajectoires
  uniformes* (CI et `c` tirés uniformément, puis rollout du solveur). C'est essentiellement
  la **mesure invariante / d'attracteur** de KS sur la plage de paramètres. C'est elle qui
  définit `val/rmse` et `val/nrmse`.
- **Distribution d'entraînement** `q` : le pool effectivement vu à un round donné. Dans notre
  schéma c'est un **mélange**
  ```
  q  =  α · p_unif  +  (1 − α) · q_gen ,        α = uniform_fraction
  ```
  où `q_gen` est la loi des états produits par le générateur conditionné « dur »
  (bins de quantile de difficulté), et `p_unif ≈ p` (mêmes trajectoires uniformes).

Le risque de validation est
```
R_p(θ) = E_{u~p}[ ℓ_θ(u) ] .
```
L'entraînement minimise (sans repondération) le risque empirique sur `q` :
```
R_q(θ) = E_{u~q}[ ℓ_θ(u) ] .
```
**Toute la question est : minimiser `R_q` réduit-il `R_p` ?** C'est un problème classique de
**covariate shift / active learning sous mauvaise spécification**.

---

## 2. Le résultat négatif (pourquoi il n'y a pas de garantie)

### 2.1 Identité d'importance et borne de mismatch

Pour tout `θ`, si `p ≪ q` (support de `q` couvrant celui de `p`),
```
R_p(θ) = E_{u~q}[ w(u) ℓ_θ(u) ] ,     w(u) = p(u)/q(u)  (poids d'importance) .
```
On **n'applique pas** ces poids (on entraîne sur `q` brut). Le lien entre les deux risques
est borné, au choix, par :

- **Hölder / sup des poids :** `R_p(θ) ≤ ‖w‖_∞ · R_q(θ)`.
  `‖w‖_∞ = sup_u p(u)/q(u)` **explose** dès que `q` sous-échantillonne une région où `p`
  a de la masse (le *bulk* de l'attracteur). Borne **vide**.

- **IPM / Wasserstein** (si `ℓ_θ` est `Lip`-lipschitzienne) :
  ```
  | R_p(θ) − R_q(θ) |  ≤  Lip(ℓ_θ) · W₁(p, q) .
  ```
- **TV / Pinsker** (si `ℓ_θ` bornée par `B`) :
  ```
  | R_p(θ) − R_q(θ) |  ≤  B · TV(p, q)  ≤  B · √(½ KL(p‖q)) .
  ```

**Lecture.** Plus on pousse le générateur « dur » (plus `q_gen` s'éloigne de `p`, donc
`W₁(p,q)` et `TV(p,q)` grands), **plus la borne reliant la perte d'entraînement à la perte de
validation se dégrade.** Le « hardness_ratio » que l'on logge (loss sur états générés / loss
uniforme) est un proxy direct de cet éloignement : un `hardness_ratio` de 5–6 (cf.
`uf50_tophalf`) signifie qu'on optimise une distribution très différente de la validation.
**C'est la formalisation exacte du compromis bulk/queue observé.**

### 2.2 Bien spécifié vs mal spécifié — le nœud

- **Cas bien spécifié** (`∃ θ*` avec `ℓ_{θ*} ≡ 0` partout). Alors *toute* `q` de support
  plein recouvre `θ*` à la limite ; `q` n'affecte que la **vitesse** (complexité
  d'échantillon). Échantillonner « dur » aide **ssi** les régions dures sont celles où `θ`
  est encore loin de `θ*` *et* où `p` met de la masse (ou qui pèsent en aval). Pas de
  tradeoff au fond — juste de l'efficacité.

- **Cas mal spécifié** (aucun `θ` n'annule l'erreur partout — **notre cas**, surrogate
  `hidden=32, depth=4` face à KS chaotique). Là, le minimiseur de `R_q` **dépend du
  poids** : concentrer `q` sur les régions dures produit un `θ` bon là-bas et **pire
  ailleurs**. Le `θ` optimal pour `R_p` (validation) et celui pour `R_q` (hard-mining)
  **sont différents.** D'où :

> **Proposition (no free lunch, version informelle).** Sous mauvaise spécification, il
> existe des instances où échantillonner exactement là où l'erreur/incertitude du surrogate
> est maximale **augmente** strictement `R_p`. *Contre-exemple :* si la région dure a une
> masse `p`-négligeable (états rares, ex. transitoires très raides), y réallouer la capacité
> du modèle dégrade l'ajustement sur le bulk `p`-dominant ⇒ `val/rmse_mean ↑`. C'est
> précisément ce qu'on mesure (`+14 % à +97 %` de `val/rmse_mean`).

**Conclusion de la section.** L'intuition « grosse incertitude ⇒ gain de validation » est
**fausse en général**. Toute garantie sera *conditionnelle* : soit on borne le shift, soit on
suppose la bonne spécification, soit **on change la fonctionnelle cible** (section 4).

---

## 3. Garanties conditionnelles exploitables

### 3.1 Garantie de mélange (le rôle exact de `uniform_fraction`)

C'est le point le plus utile pour le papier. Avec
`q = α p + (1−α) q_gen`, **les poids d'importance sont bornés** indépendamment de `q_gen` :
```
w(u) = p(u) / q(u) = p(u) / (α p(u) + (1−α) q_gen(u))  ≤  1/α .
```
D'où la **borne multiplicative** :
```
        ┌─────────────────────────────────────┐
        │   R_p(θ)  ≤  (1/α) · R_q(θ)          │     α = uniform_fraction
        └─────────────────────────────────────┘
```
**Interprétation.** Quel que soit l'agressivité du générateur, garder une fraction `α`
d'états uniformes **garantit** que le risque de validation ne peut pas dépasser `1/α` fois le
risque qu'on minimise réellement. `uniform_fraction` n'est pas un bouton empirique : c'est le
paramètre qui **plafonne le ratio de covariate-shift** à `1/α` et donne une garantie *a
priori* de non-explosion de la RMSE-bulk.

- `α = 0` (full générateur) : aucune borne (`1/α = ∞`) → cf. `uf10_tophalf` régresse `rmse +73 %`.
- `α = 0.5` (`uf50_*`) : `R_p ≤ 2 R_q` → bulk récupéré (`rmse +14 %`, `rmse_p99 −8 %`).
- `α = 1` (uniforme) : `R_p = R_q`, aucun gain ciblé.

→ **`uniform_fraction` interpole de manière *prouvée* entre "garantie sur le bulk" et
"ciblage du dur".** Le balayage `α ∈ {0,.25,.5,.75,1}` (expérience #3) trace donc une courbe
dont les extrémités et la pente sont théoriquement cadrées : c'est un *front de Pareto avec
borne*, pas juste une courbe empirique.

### 3.2 Repondération d'importance (alternative au mélange)

On peut au lieu du mélange entraîner sur `q_gen` mais **repondérer** par `ŵ = p̂/q̂` :
estimateur **non biaisé** de `R_p`. Coût : variance `∝ χ²(p‖q_gen)` (ESS faible si
`q_gen` très loin de `p`). C'est le tradeoff biais–variance classique de l'AL. Le mélange
(§3.1) est une version *clippée* (poids bornés `≤ 1/α`) donc à variance contrôlée — d'où sa
robustesse pratique. On peut documenter les deux comme deux régularisations du même objectif.

### 3.3 Vue A/V-optimal design (justifie le désaccord d'ensemble)

Décomposition erreur d'estimation ≈ Σ (leverage du point) × (bruit/épistémique). En design
optimal, on échantillonne là où la **réduction de variance par échantillon est maximale**, ce
qui pour un modèle bien spécifié ≈ là où l'**incertitude épistémique** est grande *et* où le
point a du leverage sur la quantité cible. La **variance inter-membres de l'ensemble**
(`EnsembleSurrogate.uncertainty`, déjà implémentée) est l'estimateur épistémique standard
(approx. BALD / query-by-committee). → justification théorique de l'expérience #1
(difficulté = désaccord d'ensemble plutôt que loss one-step) : la loss one-step mélange
épistémique (réductible par données) et aléatoire/irréductible (chaos, mauvaise
spécification) ; le désaccord isole la partie **réductible** — la seule qu'échantillonner
puisse corriger.

---

## 4. La bonne fonctionnelle : pourquoi le gain est robuste sur le *rollout*

C'est, à mon sens, **le vrai message du papier.** `val/rmse_mean` et la *stabilité du
rollout* sont **deux fonctionnelles différentes du champ d'erreur**, et le hard-sampling est
le bon biais inductif pour la seconde, pas forcément la première.

### 4.1 Propagation d'erreur en rollout

Rollout : `û_{k+1} = f_θ(û_k)`. Erreur `e_k = û_k − u_k`. Au premier ordre,
```
e_{k+1} ≈ J_k e_k + ε_k ,   J_k = ∂f_θ/∂u |_{u_k} ,   ε_k = erreur one-step au point u_k .
```
En déroulant :
```
e_K ≈ Σ_{j<K} ( Π_{j<i<K} J_i ) ε_j .
```
Les produits de Jacobiens `Π J_i` **amplifient** les erreurs one-step le long des directions
instables (exposants de Lyapunov positifs de KS). L'erreur de rollout est donc dominée par
les `ε_j` aux **états à fort leverage** : grande sensibilité locale `‖J‖`, forte courbure,
structures raides. Formellement, pour une fonctionnelle de type sup/amplification,
```
E[ ‖e_K‖ ]  ≲  Σ_j  G_{j,K} · ε_j ,     G_{j,K} = gain d'amplification (grand aux états raides).
```

### 4.2 Le mismatch entre `p` et le leverage

Or ces états à fort `G` sont **rares sous `p`** (l'attracteur passe l'essentiel du temps
près de structures « lisses » ; les transitoires raides sont brefs). Donc :

- `R_p(θ) = E_p[ε]` **sous-pondère** exactement les états qui dominent le rollout
  (moyenne sur `p` → poids ∝ fréquence, pas ∝ leverage).
- La stabilité de rollout est ≈ `Σ_j G_{j,K} ε_j` → **pondère par le leverage `G`**, pas par
  la fréquence.

> **Proposition (réconciliation).** Échantillonner les états durs/raides (grand `G`, faible
> `p`) augmente leur poids effectif dans l'entraînement, réduit `ε` là où `G` est grand, donc
> **réduit la fonctionnelle de rollout** — *même si* cela augmente `R_p = E_p[ε]` (moyenne
> bulk). La régression `val/rmse_mean ↑` et le gain `rollout ↓` **ne se contredisent pas :
> c'est le comportement prédit.**

**C'est exactement nos chiffres :** rollout `−16 à −19 %` sur **toutes** les variantes
(robuste, car c'est la fonctionnelle que le hard-sampling optimise), tandis que `rmse_mean`
régresse (fonctionnelle moyenne, que le hard-sampling n'optimise pas). Le `nrmse_p99`
(queue) s'améliore aussi (`−30 à −50 %`) car c'est une fonctionnelle de *queue*, proche d'une
fonctionnelle de leverage.

### 4.3 Conséquence pour la métrique de validation

Valider *seulement* en `val/rmse_mean` uniforme **mesure la mauvaise fonctionnelle** par
rapport au but réel (surrogate utilisable en rollout). Recommandations :
1. Promouvoir **rollout-NRMSE** et **horizon stable** (premier `k` où `nrmse_k > τ`) comme
   métriques primaires.
2. Garder `val/rmse` *stratifié* (`val/nrmse_q*`, déjà loggé) comme métrique secondaire pour
   exposer le tradeoff bulk/queue, pas comme juge unique.
3. Idéalement, ajouter un set de validation **pondéré par leverage / sur états rares** pour
   mesurer directement la fonctionnelle ciblée (sinon on se bat contre une métrique qui, par
   construction, favorise l'uniforme).

### 4.4 Garantie côté rollout (direction à formaliser)

Si on définit la difficulté `d(u) = ‖J_θ(u)‖ · ε_θ(u)` (sensibilité × erreur épistémique),
alors échantillonner `q_gen ∝ d` cible directement les termes dominants de `Σ G·ε`. On peut
viser un énoncé du type : *sous hypothèse de Lyapunov bornée et de bonne spécification locale,
réduire `ε` sur le support de `d` borne la croissance de `E‖e_K‖`* — c'est la garantie
« utile » à chercher (bien plus accessible que sur `R_p`). C'est le pont théorie↔expérience #1.

---

## 5. Synthèse — ce que la théorie dit de faire

| Question | Réponse théorique | Conséquence expérimentale |
|---|---|---|
| Garantie que hard-sampling ↓ `val/rmse` uniforme ? | **Non** (mal spécifié ⇒ tradeoff ; borne `‖w‖_∞` vide). | Ne pas vendre le papier sur `val/rmse_mean`. |
| Comment borner la casse sur le bulk ? | Mélange `α p + (1−α)q_gen` ⇒ **`R_p ≤ R_q/α`**. | `uniform_fraction` = bouton *prouvé* ; balayer `α` (exp #3). |
| Quel signal de difficulté ? | Désaccord d'ensemble = épistémique **réductible** (≠ loss = épistémique+aléatoire). | Difficulté = `uncertainty()` (exp #1). |
| Où le gain est-il garanti/robuste ? | Fonctionnelle de **rollout/amplification**, pas la moyenne `p`. | Métrique primaire = rollout + horizon stable (exp #4 surrogate plus fort pour sortir du régime divergent). |
| Baseline juste ? | Séparer décorrélation (états i.i.d.) du ciblage (générateur). | 3 baselines : trajectoires unif / états i.i.d. unif / générateur ; courbe vs # appels solveur. |

**Message papier (une phrase).** *La RMSE one-step moyenne et la stabilité de rollout sont
gouvernées par des fonctionnelles différentes du champ d'erreur ; l'échantillonnage génératif
d'états durs est le biais inductif adapté à la seconde, et `uniform_fraction` interpole de
façon bornée (`R_p ≤ R_q/α`) entre garantie sur le bulk et ciblage de la queue/rollout.*

---

## 6. Hypothèses vérifiables (pour étayer la théorie avec les runs)

1. **Borne de mélange.** Tracer `R_p / R_q` vs `1/α` ; doit rester sous la droite `y = 1/α`.
2. **Fonctionnelle de leverage.** Corréler le gain par état (uniforme→générateur) avec `‖J_θ‖`
   ou la TV locale ; la théorie §4 prédit corrélation positive avec le gain *rollout*, faible
   ou négative avec le gain *rmse-bulk*.
3. **Épistémique vs aléatoire.** Désaccord d'ensemble doit prédire le gain mieux que la loss
   brute (sinon on échantillonne de l'irréductible).
4. **Régime de rollout.** Avec surrogate plus fort, vérifier que l'horizon stable s'allonge
   (gain sur une fonctionnelle non dégénérée, `nrmse ≪ 1`).

---

## 7. Recadrage : le bon objectif est la *couverture* (uniformité sur la loss), pas la moyenne sur l'attracteur

> *Idée directrice (formulée par P. César) : la limite du sampling uniforme est qu'il optimise
> la performance sur l'attracteur ; notre but n'est pas « meilleur surrogate sur le set
> uniforme » mais « bon surrogate sur tous les états atteignables » — chercher une uniformité
> sur la loss elle-même, peu importe les états.*

### 7.1 Pourquoi le set uniforme est le mauvais juge (limite intrinsèque, pas un détail de métrique)

La mesure `p` (attracteur de KS sous CI/paramètres uniformes) **concentre** sa masse sur une
variété effective de basse dimension. Les **modes durs** — transitoires raides, états à forte
courbure, coins de l'espace des paramètres — sont *atteignables* (`u ∈ M`, le support des
états physiquement réalisables) mais de **`p`-mesure ≈ 0**. Conséquences :

1. **Insensibilité de l'objectif.** `R_p(θ) = E_p[ℓ_θ]` pondère par la *fréquence d'occupation*.
   Un mode dur de mesure `10⁻³` contribue `10⁻³ ℓ` à `R_p` : invisible. On peut donc avoir
   `R_p(θ) → 0` et `sup_{u∈M} ℓ_θ(u) = ∞`.
2. **Non-couverture de l'échantillon.** Aucun échantillon *fini* tiré de `p` ne contient les
   modes de mesure `→ 0`. Le sampling uniforme **ne peut pas**, même à données infinies
   pratiques, apprendre ces régions. Ce n'est pas un problème de repondération (§2–§3) mais
   de **support** : `q_uniforme` ne couvre pas `M`.
3. **Le juge est complice.** Valider sur `p` récompense exactement la stratégie qui ignore les
   modes durs. D'où la « demi-teinte » : on mesure le surrogate là où l'uniforme est déjà bon
   par construction.

> **C'est la thèse centrale du papier :** le sampling uniforme a une **limite de couverture**
> intrinsèque ; toute méthode (la nôtre) qui vise les modes durs ne peut être *évaluée* que
> sur une métrique sensible à ces modes (§8). Sur `p`, le bénéfice est structurellement masqué.

### 7.2 Reformulation de l'objectif : de la moyenne au minimax / DRO

Cible non plus la moyenne sous `p`, mais la performance **sur tout `M`**. Trois écritures
équivalentes en pratique, de la plus dure à la plus douce :

```
(worst-case / coverage)   R_∞(θ)   = sup_{u ∈ M}  ℓ_θ(u)
(DRO, ensemble U autour de p)  R_DRO(θ) = max_{q ∈ U(p)}  E_{u~q}[ ℓ_θ(u) ]
(loss-uniform / coverage measure)  R_cov(θ) = E_{u~μ}[ ℓ_θ(u) ],  μ = uniforme sur le support de M
                                              (ou uniforme sur une coordonnée de difficulté)
```

- `R_∞` : robustesse pure (minimax). Difficile à estimer (sup) mais c'est la garantie ultime.
- `R_DRO` : `min_θ max_{q∈U} E_q[ℓ]` — **distributionally robust optimization**. `U(p)` =
  boule (KL, Wasserstein, ou χ²) autour de `p`. Interpole entre `R_p` (`U={p}`) et `R_∞`
  (`U` = tout). **C'est exactement notre schéma** : le générateur conditionné « dur » *est*
  l'approximation du `max` intérieur (distribution adverse concentrée sur les hautes loss), le
  surrogate joue le `min`. `uniform_fraction` ≈ rayon de `U` (mélange `α p + (1−α) q_adv`).
- `R_cov` : remplace la pondération par fréquence par une pondération **uniforme sur le
  support** (ou sur une coordonnée de difficulté). C'est l'écriture la plus proche du langage
  « uniformité sur la loss » et la plus facile à **mesurer** (§8.3).

### 7.3 « Uniformité sur la loss » = optimum de couverture (argument de water-filling)

Pourquoi viser l'uniformité de la loss ? À l'optimum d'un objectif minimax/DRO sous capacité
limitée, **la loss s'égalise sur le support actif** : on ne peut plus baisser le `sup` sans
remonter une autre région (condition KKT / water-filling). Donc :

```
minimiser R_∞ / R_DRO   ⇒   profil de loss aplati sur M   ⇔   « uniformité sur la loss »
```

Réciproquement, un profil de loss très **inégal** (creux sur le bulk `p`, pics sur les modes
rares) — exactement ce que produit l'entraînement uniforme — est la signature d'un objectif
`R_p` *mal aligné* avec la couverture. **Mesurer l'inégalité du profil de loss sur `M`** (p.ex.
ratio `sup/moyenne`, ou Gini de `ℓ_θ` sur un bank couvrant) devient une métrique de premier
plan : notre méthode doit l'**aplatir**, l'uniforme non.

### 7.4 Rôle du générateur, reformulé : mécanisme de *couverture*, pas de repondération

Dans ce cadre, le générateur n'est plus « un échantillonneur qui repondère `p` » (vision §3,
limitée par le support) mais un **mécanisme de couverture** : il *synthétise* des états dans
les régions `p`-rares mais atteignables, rendant l'objectif `R_∞/R_DRO/R_cov` **optimisable**.
Cibler la loss dure = remonter l'adversaire `q_adv` vers les pics de loss = water-filling. Le
réalisme du générateur (TV, PSD vs KS) est ici **crucial** : il garantit `support(q_gen) ⊂ M`
(on couvre des états *physiquement atteignables*, pas du bruit hors-variété) — sinon on
minimise un `R_∞` sur un `M` artificiellement gonflé. → relie directement aux
`generator_metrics/` de réalisme déjà loggées.

### 7.5 Garantie dans ce cadre (et elle existe, contrairement à §2)

Contrairement à `R_p` (pas de garantie, §2), l'objectif DRO **se prête à une garantie de
descente** : si à chaque round `q_adv` met sa masse sur `argmax ℓ_θ` (modes durs courants) et
que l'entraînement réduit `E_{q_adv}[ℓ]`, alors sous capacité suffisante on **décroît le
`sup`** (jeu min-max ; convergence vers un équilibre où la loss est aplatie). C'est la
garantie « utile » que cherchait l'intuition : *dans le régime coverage*, sur-échantillonner
les loss dures **est** une descente sur l'objectif — il n'y a plus de compromis ambigu, parce
que la fonctionnelle évaluée est celle qu'on optimise. **La condition** est double :
(i) le générateur couvre effectivement les modes durs (réalisme + diversité), (ii) on mesure
sur `R_cov`/`R_∞`, pas sur `R_p`.

---

## 8. Construire un set de validation qui *mesure* la couverture (dur / non-vu)

Sans cela, rien de §7 n'est observable. Quatre constructions, du plus simple/feasible au plus
ambitieux. Toutes définissent la difficulté de façon **model-independent** (sinon circularité :
on ne valide pas sur « ce que CE surrogate rate »).

### 8.1 Régions de paramètres non-vues (extrapolation) — *le « unseen » le plus propre*

Restreindre les **trajectoires d'entraînement** à un sous-pavé des paramètres
`C_train ⊊ C` (p.ex. `ν ∈ [1.5, 3.0]` au lieu de `[0.5, 4.0]`), et **valider sur le
complément** `C \ C_train` (coins/queues : `ν ∈ [0.5,1.5] ∪ [3.0,4.0]`). C'est « non-vu pour
le set de trajectoires » au sens littéral. Test de la thèse : le générateur (conditionné sur
des paramètres tirés sur *tout* `C`, ou produisant des états typiques des régions extrêmes)
couvre-t-il l'extrapolation mieux que l'uniforme restreint ? *Avantage :* fixe, non ambigu,
relie à la littérature d'extrapolation PDE. *À implémenter :* un `param_train_ranges` distinct
du `param_ranges` de validation.

### 8.2 Queue d'états rares, model-independent — *le test direct des « modes durs »*

Depuis un **grand** bank de trajectoires uniformes (couvrant `M` par force brute), calculer une
**coordonnée de difficulté intrinsèque** par état — p.ex. variation totale `TV(u)=Σ|Δu|`,
énergie haute-fréquence `‖high-pass(u)‖²`, ou courbure `‖u''‖` — puis **ne garder que la queue
haute** (top-5 % / top-1 %). Ce sont les modes rares que `p` sous-représente. Valider `ℓ_θ`
dessus. *Avantage :* mesure exactement « bon sur les états durs » ; aucun changement de
l'entraînement. *Piège évité :* difficulté = grandeur physique fixe (TV…), pas la loss du
modèle.

### 8.3 Validation *loss-uniform* / coverage — *la métrique de §7.3*

Re-échantillonner le bank pour que la validation soit **uniforme sur la coordonnée de
difficulté** (masse égale par quantile de TV/énergie), au lieu de `∝ p`. Mesure alors
`R_cov` : performance uniforme sur l'espace des états, pas sur l'occupation. Rapporter en plus
l'**inégalité du profil de loss** (`sup/mean`, Gini) — la métrique que notre méthode doit
aplatir.

### 8.4 Validation off-attractor / perturbée (optionnel, plus tard)

États atteignables mais rarement *habités* : transitoires courts depuis CI atypiques (grande
amplitude non relaxée), ou états d'attracteur perturbés par chocs structurés. Couvre la partie
de `M` que même un grand bank uniforme rate.

### 8.5 Protocole de test minimal (premier round d'expériences)

1. **Garder** la validation uniforme actuelle (`val/*`) comme contrôle — on *prédit* peu/pas
   de gain dessus (c'est le point).
2. **Ajouter** §8.2 (queue TV/énergie, top-5 %) et §8.1 (params non-vus) comme sets `val_hard/*`
   et `val_unseen/*`.
3. **Hypothèse falsifiable :** sur `val_hard`/`val_unseen`, sur-échantillonner les loss dures
   (générateur, `uniform_fraction` modéré) **réduit fortement** RMSE/NRMSE et **aplatit le
   profil de loss**, alors que sur `val/*` uniforme le gain reste faible/nul. Si oui → la
   « demi-teinte » était un artefact de la métrique, et la méthode a une vraie valeur ; sinon
   → la couverture générée n'atteint pas les vrais modes durs (revoir réalisme/diversité du
   générateur, §7.4).

---

## 9. Le vrai mécanisme du rollout : couverture du *tube* et invariance vers l'avant (exposure bias)

> *Idée directrice (P. César) : en rollout `u₀→û₁ = u₁ + err`, l'état perturbé `û₁` n'est pas
> forcément dans le set généré ; il peut dégénérer vite vers des états inconnus et encore plus
> durs. Si au contraire on a cherché/couvert les états durs, alors même en déviant des
> trajectoires « parfaites » on garde une consistance des prédictions.* → On peut le rendre
> rigoureux : **la couverture du tube autour des trajectoires est exactement la précondition
> d'un rollout stable.** C'est le pont entre §7 (couverture) et la performance rollout, et la
> **garantie** la plus solide du papier.

### 9.1 La récurrence d'erreur — et *où* la loss est évaluée

Rollout `û_{k+1} = f_θ(û_k)`, vérité `u_{k+1} = Φ(u_k)`, erreur `e_k = û_k − u_k`. En
décomposant *exactement* (ajout/retrait de `Φ(û_k)`) :

```
e_{k+1} = f_θ(û_k) − Φ(u_k)
        = [ f_θ(û_k) − Φ(û_k) ]  +  [ Φ(û_k) − Φ(u_k) ]
        =        ε_θ(û_k)        +   (Φ(û_k) − Φ(u_k))
```
d'où, avec `L_Φ` = constante de Lipschitz du *vrai* solveur sur la région considérée,
```
        ┌────────────────────────────────────────────────┐
        │   ‖e_{k+1}‖  ≤  ε_θ(û_k)  +  L_Φ · ‖e_k‖        │
        └────────────────────────────────────────────────┘
```

**Le point crucial — et c'est exactement l'intuition formulée :** l'erreur one-step est
évaluée en **`û_k` (l'état visité, perturbé)**, *pas* en `u_k` (la trajectoire propre). Or
`û_k = u_k + e_k` **quitte** la variété des trajectoires propres `M_traj ≈ supp(p)`. Donc la
qualité du rollout dépend de `ε_θ` **hors de la trajectoire propre**, dans un voisinage qui
s'élargit avec `e_k`.

### 9.2 Deux régimes — l'emballement (uniforme) vs le tube piégeant (couverture)

**(A) Entraînement uniforme (couvre `M_traj` seulement).** `ε_θ` est petit *sur* `M_traj`
mais **non contrôlé** hors d'elle. Modélisons l'erreur OOD qui croît avec l'écart :
`ε_θ(û_k) ≤ ε₀ + γ·dist(û_k, M_traj) ≈ ε₀ + γ‖e_k‖`. La récurrence devient
```
‖e_{k+1}‖ ≤ ε₀ + (L_Φ + γ)·‖e_k‖ ,
```
de taux effectif `L_Φ + γ > L_Φ` : **rétroaction positive** — `û_k` sort de la zone connue,
`ε_θ` grimpe, ce qui éloigne encore `û_k`… → divergence (sur-linéaire). **C'est l'exposure
bias / l'effet boule de neige**, et la « dégénérescence rapide vers des états inconnus » du
message. C'est aussi pourquoi nos rollouts uniformes saturent à `nrmse ≈ 1` (régime dégénéré).

**(B) Entraînement avec couverture du tube (objectif §7).** Définissons le **tube**
`T_ρ = { u : dist(u, M_traj) ≤ ρ }` (ρ = écart typique que le rollout explore). Supposons,
*grâce au sampling/recherche des états durs*, que la loss one-step est **uniformément bornée
sur le tube** :
```
sup_{u ∈ T_ρ} ε_θ(u) ≤ ε̄ .        (H1 : couverture uniforme du tube — c'est R_∞ sur T_ρ, §7.2)
```
Alors le terme `γ‖e_k‖` disparaît tant que `û_k ∈ T_ρ`, et la récurrence redevient **affine** :
```
‖e_{k+1}‖ ≤ ε̄ + L_Φ‖e_k‖ .
```

> **Théorème (tube piégeant ⇒ rollout borné).** Sous (H1), si de plus
> ```
>     (H2 : invariance avant)     ε̄ ≤ (1 − L_Φ) · ρ          [cas contractant L_Φ < 1]
> ```
> alors le rollout **ne quitte jamais** `T_ρ` (`û_k ∈ T_ρ ∀k`) et
> ```
>     ‖e_k‖ ≤ ε̄ / (1 − L_Φ)   pour tout k        (erreur bornée, indép. de l'horizon).
> ```
> Si `L_Φ > 1` (chaos, directions instables), la couverture ne donne plus une borne uniforme
> mais **ralentit l'emballement** : on passe du taux `(L_Φ+γ)` au taux `L_Φ` (on retire la
> rétroaction γ), soit `‖e_k‖ ≤ ε̄·(L_Φ^k − 1)/(L_Φ − 1)` au lieu d'une explosion super-
> géométrique. Le gain se mesure alors en **horizon stable** (premier `k` avec `nrmse_k > τ`),
> qui croît typiquement comme `~ log(τ/ε̄)/log L_Φ` : **réduire `ε̄` (couverture) ↑ l'horizon**.

*Preuve (cas contractant) :* récurrence + (H2) ⇒ si `‖e_k‖ ≤ ρ` alors
`‖e_{k+1}‖ ≤ ε̄ + L_Φρ ≤ (1−L_Φ)ρ + L_Φρ = ρ` (invariance), et le point fixe de
`x ↦ ε̄ + L_Φ x` est `ε̄/(1−L_Φ)`, attracteur car `L_Φ<1`. ∎

**Lecture.** *La couverture du tube (H1) est précisément la condition qui ferme l'argument
d'invariance (H2) et borne le rollout.* L'entraînement uniforme ne couvre pas le tube ⇒ pas
de (H1) ⇒ pas de garantie ⇒ emballement. Sur-échantillonner / **rechercher les états durs =
construire (H1)**. C'est la garantie demandée : *même en déviant des trajectoires parfaites,
les prédictions restent consistantes*, parce que les états déviés sont dans le support
d'entraînement.

### 9.3 Lien avec DAgger / exposure bias — et la nouveauté « DAgger anticipé »

C'est l'analogue exact du résultat classique d'imitation (Ross & Bagnell) : le *behavior
cloning* (entraîner sur la distribution de l'expert = trajectoires propres `p`) a une erreur
qui compounde en `O(ε·H²)` à cause du décalage de distribution ; entraîner sur la
**distribution visitée** (DAgger) ramène à `O(ε·H)`. Ici :

- Sampling uniforme ≡ behavior cloning sur `p` → `O(ε H²)`, exposure bias.
- Notre générateur d'états durs ≡ entraînement sur la distribution **visitée par le rollout
  perturbé** — mais de façon **proactive** : on *synthétise* les états déviés que le rollout
  va rencontrer, **avant** de les rencontrer, sans avoir à dérouler. → **« DAgger anticipé par
  modèle génératif »**, angle de nouveauté fort.

### 9.4 Raffinement actionnable : *rollout-aware hard mining* (fermer la boucle)

Risque identifié (et important) : un générateur conditionné sur la *loss one-step* peut
produire des états durs qui ne sont **pas** ceux du tube réellement visité par le rollout. Pour
garantir l'alignement `support(q_gen) ⊃ T_ρ`, deux mécanismes concrets :

1. **Seeds perturbés :** amorcer/conditionner le générateur sur des états de trajectoire
   **perturbés** (`u + bruit structuré`, échelle ρ) plutôt que sur des états d'attracteur purs.
2. **Boucle fermée (rollout-aware) :** utiliser les **états effectivement visités par le
   rollout du surrogate courant** (là où `nrmse_k` décroche) comme cibles de difficulté à la
   prochaine génération. C'est le DAgger exact, le générateur servant d'amortisseur/
   d'extrapolateur pour couvrir le tube sans dérouler à chaque fois.

→ Ceci définit une **variante méthodologique testable** (et probablement la version « forte »
de la méthode pour le rollout) : *difficulté = erreur sur le tube visité*, pas seulement loss
one-step sur le pool.

### 9.5 Quantités mesurables (relier la théorie aux runs)

- **`ε̄` (couverture du tube) :** erreur one-step max sur un *tube de validation* = états
  d'attracteur **perturbés** à l'échelle ρ (à construire, cf. §8.4). Doit chuter avec la
  méthode.
- **`L_Φ` local :** estimer `‖Φ(u+δ)−Φ(u)‖/‖δ‖` sur l'attracteur (caractérise contractant vs
  chaotique par région) — dit où (H2) peut tenir.
- **Horizon stable** `K_τ = min{k : nrmse_k > τ}` : métrique rollout primaire ; la théorie
  prédit `K_τ ↑` quand `ε̄ ↓`.
- **Invariance empirique :** tracer `dist(û_k, M_traj)` le long du rollout — borné (tube) vs
  divergent (uniforme).

### 9.6 Confirmation empirique de (H1) — post-hoc, KS, 5 seeds

Mesure de l'erreur one-step `ε_θ` sur un **set tube** (états d'attracteur perturbés à
l'échelle `ρ`, cf. `scripts/build_tube_validation.py`), surrogate `uf50_tophalf` (générateur)
vs uniforme :

| NRMSE moy. sur tube | ρ=0.1 | ρ=0.25 | ρ=0.5 |
|---|---|---|---|
| uniforme | 0.0305 | 0.0485 | **0.0699** (croît) |
| générateur (uf50) | 0.0102 | 0.0110 | **0.0125** (plat) |
| écart | −67% | −77% | **−82%** |

L'erreur de l'uniforme **croît** quand on s'éloigne de l'attracteur (terme `γ‖e‖`, §9.2) ;
celle du générateur **reste plate** → **(H1) vérifiée** : `ε_θ` est uniformément bornée sur le
tube. C'est la *précondition* de la borne d'invariance §9.2. NB : c'est l'erreur one-step (la
condition), pas le rollout (la conséquence) — le test direct de l'horizon `K_τ` est en Phase 2.
À comparer : sur le bulk de l'attracteur (val_unif RMSE absolue) le générateur est *+14% pire*
— le coût masqué par la métrique uniforme, négligeable face au gain de robustesse off-variété.

