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

*Phases suivantes : en cours*
