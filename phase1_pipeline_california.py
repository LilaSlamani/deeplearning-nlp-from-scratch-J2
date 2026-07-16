import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Choix : split d'abord, scaler fitté sur X_train uniquement -> evite le data leakage

housing = fetch_california_housing()
X, y = housing.data, housing.target  # 20 640 maisons, 8 features, cible continue

# Split 1 : 80% train+val / 20% test (le test ne sera touche qu'a la fin)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split 2 : 80% train / 20% val (val surveille l'entrainement sans y participer)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalisation : centre-reduit chaque feature -> reseau apprend plus facilement
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)   # calcule les stats sur train
X_val_norm   = scaler.transform(X_val)         # applique les memes stats
X_test_norm  = scaler.transform(X_test)        # applique les memes stats

# Verification des shapes
print(f"X_train shape : {X_train_norm.shape}")  # attendu : (13209, 8)
print(f"X_val shape   : {X_val_norm.shape}")    # attendu : (3303, 8)
print(f"X_test shape  : {X_test_norm.shape}")   # attendu : (4128, 8)

# Verification normalisation : mean ~ 0, std ~ 1 sur le train
print(f"\nX_train_norm mean : {X_train_norm.mean(axis=0).round(4)}")
print(f"X_train_norm std  : {X_train_norm.std(axis=0).round(4)}")

# Verification du chargement : 8 features attendues
print(f"\nFeatures ({len(housing.feature_names)}) : {list(housing.feature_names)}")
