# Jour 2 - PMC sur données structurées

Pipeline complet : chargement, normalisation, entraînement, diagnostic TensorBoard, tuning, régularisation.  
Trois datasets : California Housing (régression), Pima Diabetes (classification binaire), Wine Quality (multiclassification).

---

## Phase 1 - Pipeline de données (California Housing)

**Fichier :** `phase1_pipeline_california.py`  
**Dataset :** California Housing - 20 640 maisons, 8 features numériques, cible continue (prix en $100k)  
**Objectif :** split train/val/test propre + normalisation sans data leakage

### Scénario normal

| Ensemble | Shape | Mean (après norm) | Std (après norm) |
|---|---|---|---|
| X_train | (13209, 8) | ~0.0 | ~1.0 |
| X_val | (3303, 8) | - | - |
| X_test | (4128, 8) | - | - |

Features : `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`

### Cas limite - Data leakage

Comparaison de la moyenne du test set selon l'ordre fit/split :

| Pipeline | X_test mean (par feature) |
|---|---|
| Correct (fit sur X_train uniquement) | `[-0.0204  0.0157 -0.0077  0.006  -0.0047 -0.0093 -0.0282  0.0319]` |
| Bugué (fit sur X entier) | `[-0.0212  0.0099 -0.0101 -0.0001 -0.0034 -0.0101 -0.0211  0.0251]` |

**Observation :** le pipeline bugué a des moyennes légèrement plus proches de 0 sur le test set (ex. colonne AveBedrms : 0.006 vs -0.0001). L'effet est discret car le dataset est grand (20 640 exemples) - sur un petit dataset l'écart serait bien plus visible. C'est pourquoi le data leakage est dangereux : il ne saute pas aux yeux sur les métriques finales.

### Scénario adversarial - Donnée hors distribution

Entrée : `MedInc=99999, HouseAge=-99999` (valeurs aberrantes impossibles en production normale)

| Feature | Valeur brute | Valeur normalisée |
|---|---|---|
| MedInc | 99999 | **52912.40** |
| HouseAge | -99999 | -7945.60 |

**Observation :** pendant l'entraînement, les valeurs normalisées de MedInc restent entre -2 et +3. Le réseau reçoit ici 52 912 comme entrée : il n'a rien appris dans cette zone, sa prédiction est non fiable. c'est un problème en production : toute valeur normalisée > 10 indique une donnée hors distribution.

---

## Phase 2 - Baseline PMC régression (California Housing)

**Fichier :** `phase2_baseline_regression.py`  
**Objectif :** premier modèle Keras de régression, 100 epochs, lecture des métriques epoch par epoch

### Scénario normal

| Métrique | Attendu (cours) | Obtenu |
|---|---|---|
| MAE test | 0.5 à 0.7 | **0.3585** (35 850 $ d'erreur moyenne) |
| val_loss | descend et se stabilise | stable autour de 0.28-0.29 |
| Léger overfitting fin | possible | val_loss remonte légèrement epochs 98-100 |

Architecture : Dense(64, relu) -> Dense(32, relu) -> Dense(1) sans activation  
Mieux qu'attendu : probablement dû aux optimisations TF 2.21.

### Cas limite - Impact du batch_size (10 epochs)

| batch_size | Mises à jour/epoch | val_loss epoch 10 |
|---|---|---|
| 1 (SGD pur) | 13 209 | **0.3795** |
| 13 209 (Batch GD) | 1 | 4.6929 |

SGD pur "gagne" car il fait 132 090 mises à jour vs 10 pour Batch GD. La comparaison n'est pas équitable en nb d'epochs. En pratique : batch=32 est le bon compromis vitesse/stabilité.

### Scénario adversarial - Entraînement sans normalisation

| | val_loss epoch 1 | val_loss epoch 10 |
|---|---|---|
| Sans normalisation | **1.90** (loss train = 113.87) | 0.99 (oscillations) |
| Avec normalisation | ~0.85 | ~0.28 |

La loss train démarre à 113.87 à l'epoch 1. Puis oscillations chaotiques (11.14 -> 3.64 -> 7.32) car Latitude/Longitude (~37-122) déséquilibrent les gradients et déstabilisent Adam. La normalisation n'est pas optionnelle.

---

## Phase 3 - TensorBoard (California Housing)

**Fichier :** `phase3_tensorboard_california.py`  
**Objectif :** comparer visuellement l'impact de la normalisation via deux runs TensorBoard horodatés

### Scénario normal

| Run | val_loss (tendance) | Comportement |
|---|---|---|
| california_norm | stable ~0.3 - 0.4 | convergence propre et régulière |
| california_raw | oscille entre 0.5 et 5+ | dégradé, instable tout le long |

Logs générés dans `logs/fit/` avec timestamp (format `HHMMSS`).  
Commande pour visualiser : `tensorboard --logdir=logs/fit`

### Cas limite

Logdir inexistant (`tensorboard --logdir=logs/fit_vide`) : TensorBoard démarre sans erreur mais affiche "No dashboards are active". Piège courant : on croit que l'entraînement a raté alors que c'est juste le chemin qui est faux.

### Scénario adversarial

Deux instances TensorBoard sur le même port 6006 : la seconde refuse de démarrer avec "address already in use". Solution Windows : fermer le premier terminal.

### Vérification automatique

```
Runs california_norm trouvés : 3
Runs california_raw  trouvés : 1
```

---

## Phase 4 - Baseline PMC classification binaire (Pima Diabetes)

**Fichier :** `phase4_pima_baseline.py`  
**Dataset :** Pima Indians Diabetes - 768 patientes, 8 features médicales, cible binaire (0/1)  
**Objectif :** premier modèle de classification binaire, sigmoid + binary_crossentropy

### Scénario normal

| Métrique | Valeur |
|---|---|
| Meilleure val_accuracy | **0.7724** (epoch 5-8) |
| val_accuracy epoch 100 | 0.7480 |
| accuracy train epoch 100 | 0.9063 |
| Moyenne des prédictions | 0.3858 (pas de collapse sur classe 0) |

Architecture : Dense(64, relu) -> Dense(32, relu) -> Dense(1, sigmoid) - 2 689 paramètres

**Overfitting prononcé** : le meilleur score de validation est atteint dès l'epoch 5-8, puis la val_loss remonte de 0.48 à 0.54 tandis que la loss train continue de descendre à 0.26. Écart de 16 points entre train accuracy (91%) et val_accuracy (75%) à l'epoch 100. Phase 5 (régularisation) corrige exactement ça.

### Cas limite - Entrée hors distribution

Entrée : `Glucose=-99999, BMI=0` (valeurs impossibles)

| Sortie sigmoid | Interprétation |
|---|---|
| **1.0000** | Modèle sûr à 100% que la patiente est diabétique |

La sigmoid respecte la contrainte [0,1] mais extrapole sans garde-fou. En production, il faut valider les entrées avant d'appeler `predict()`.

### Scénario adversarial - relu à la sortie au lieu de sigmoid

| Activation sortie | val_accuracy (10 epochs) |
|---|---|
| sigmoid (correct) | **0.7642** |
| relu (bug) | 0.7561 |

Différence faible (0.008) car sur Pima les logits restent majoritairement positifs. Le bug relu est plus visible sur des datasets où les logits sont souvent négatifs.

---

## Phase 5 - Régularisation (Pima Diabetes)

**Fichier :** `phase5_pima_regularisation.py`  
**Objectif :** comparer baseline, L2, et L2+Dropout sur overfit/convergence

### Scénario normal - 3 configurations

| Config | Epoch arrêt | Val_accuracy max | Comportement courbes |
|---|---|---|---|
| Baseline | 42 | 0.7724 | écart train/val visible, divergence progressive |
| L2 seul | 90 | 0.7642 | courbes collées plus longtemps, convergence plus lente |
| L2 + Dropout | 106 | **0.7805** | courbes quasi confondues tout le long |

L2 + Dropout donne la meilleure val_accuracy ET les courbes les plus saines (train et val restent proches). Graphe disponible : `phase5_pima_3configs.png`.

### Cas limite (qualité tests)

L2 réduit l'overfit de 75% sur 100 epochs :

| Config | Écart train-val (loss) | Interprétation |
|---|---|---|
| Baseline | -0.3325 | val_loss dépasse train_loss de 0.33 = overfit fort |
| L2 (0.01) | -0.0823 | val_loss dépasse train_loss de 0.08 = overfit réduit |

Dropout désactivé en inférence : confirmé (deux appels `predict()` identiques).

### Scénario adversarial - L2 lambda=10

val_accuracy = **0.6260** (en dessous du seuil de la classe majoritaire à 65%). Lambda trop fort écrase les poids à zéro, le modèle fait pire qu'une règle naïve.

---

## Phase 7 - Baseline PMC multiclassification (Wine Quality)

**Fichier :** `phase7_wine_baseline.py`  
**Dataset :** Wine Quality UCI - 1 599 vins rouges, 11 features chimiques, 3 classes (low/medium/high)  
**Objectif :** premier modèle de multiclassification, softmax + sparse_categorical_crossentropy

### Scénario normal

| Métrique | Valeur |
|---|---|
| Val_accuracy finale (epoch 100) | 0.8555 |
| Test accuracy | **0.8594** |
| Val_accuracy max | 0.8633 (epochs 81 et 94) |

Architecture : Dense(64, relu) -> Dense(32, relu) -> Dense(3, softmax) - 2 947 paramètres

Overfitting tardif : val_loss minimale à l'epoch 22 (0.4573), remonte à 0.59 à l'epoch 100 pendant que la train loss descend à 0.14. Moins agressif qu'en phase 4 grâce au dataset plus grand (1 279 exemples en train). Le test accuracy dépasse la val_accuracy finale : bonne généralisation.

Distribution classes : medium ~82%, high ~14%, low ~4%. Le modèle atteint 77% dès l'epoch 1 en apprenant rapidement la classe majoritaire.

### Cas limite - stratify

| Split | Classe 0 (low) dans train |
|---|---|
| Sans stratify (random_state=42) | 52 exemples |
| Avec stratify | 50 exemples |

Écart faible ici (grand dataset, seed favorable). Sur un petit dataset ou un autre seed, sans stratify la classe low peut disparaître complètement du train.

### Scénario adversarial - sigmoid au lieu de softmax

| Activation | Exemple : somme des 3 sorties |
|---|---|
| softmax (correct) | **1.0000** (toujours) |
| sigmoid (bug) | 1.68, 1.42, 1.74 (somme > 1) |

Avec sigmoid, les 3 neurones de sortie sont indépendants et ne se coordonnent pas. Les sorties ne sont plus des probabilités, elles ne peuvent pas être interprétées comme telles.

---

## Phase 6 - Keras Tuner RandomSearch (Pima Diabetes)

**Fichier :** `phase6_pima_kerastuner.py`  
**Objectif :** trouver automatiquement les meilleurs hyperparamètres via 15 trials aléatoires

### Top 5 trials

| Trial | units_1 | units_2 | activation | dropout | lr | val_accuracy |
|---|---|---|---|---|---|---|
| 05 | 32 | 48 | tanh | 0.4 | 0.0005 | **0.7886** |
| 08 | 32 | 32 | tanh | 0.0 | 0.005 | **0.7886** |
| 10 | 64 | 64 | tanh | 0.3 | 0.0005 | **0.7886** |
| 03 | 32 | 48 | relu | 0.1 | 0.001 | 0.7805 |
| 06 | 32 | 32 | tanh | 0.2 | 0.0005 | 0.7805 |

**Best model val_accuracy (réentraîné 200 epochs) : 0.7805**

### Invariants observés

- **tanh** : 4 trials sur 5. Sur Pima (données normalisées, centrées), tanh surpasse relu.
- **lr=0.0005** : 4 trials sur 5. Learning rate faible = convergence plus stable sur petit dataset.
- **Pas d'invariant sur dropout/units** : les 3 meilleurs ont le même score malgré des architectures très différentes. Pima est trop petit pour discriminer ces variations.

### Comparaison des phases sur Pima

| Phase | Technique | Val_accuracy max |
|---|---|---|
| Phase 4 | Baseline | 0.7724 |
| Phase 5 | L2 + Dropout | 0.7805 |
| Phase 6 | Keras Tuner | **0.7886** |

### Cas limite (qualité tests)

seed=42 : deux tuners indépendants avec 3 trials donnent exactement les mêmes hyperparamètres (`Identiques : True`). Espace de recherche minimal (une seule valeur) : tous les trials ont `u=32`, le tuner tourne sans vraiment chercher.

---

*Phases suivantes : en cours*
