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

**Observation :** pendant l'entraînement, les valeurs normalisées de MedInc restent entre -2 et +3. Le réseau reçoit ici 52 912 comme entrée : il n'a rien appris dans cette zone, sa prédiction est non fiable. Signal d'alarme en production : toute valeur normalisée > 10 indique une donnée hors distribution.

---

*Phases suivantes : en cours*
