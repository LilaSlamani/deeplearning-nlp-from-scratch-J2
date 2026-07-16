import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Pipeline de reference (identique a phase1)
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm   = scaler.transform(X_val)
X_test_norm  = scaler.transform(X_test)

# === CAS LIMITE : data leakage ===
# Si on fitte le scaler sur X entier, le test set "participe" aux stats -> fuite
scaler_bad = StandardScaler()
scaler_bad.fit(X)  # bug : X entier au lieu de X_train
X_test_norm_bad = scaler_bad.transform(X_test)

print("=== CAS LIMITE : data leakage ===")
print(f"Pipeline correct -> X_test mean : {X_test_norm.mean(axis=0).round(4)}")
print(f"Pipeline bugge   -> X_test mean : {X_test_norm_bad.mean(axis=0).round(4)}")
# Attendu : le pipeline bugge a une moyenne plus proche de 0 (le test a "vu" ses propres stats)

# === ADVERSARIAL : donnee hors distribution ===
# Valeur aberrante : revenu = 99999, age = -99999 (impossible en vrai)
X_extreme = np.array([[99999, -99999, 0, 0, 0, 0, 37.0, -120.0]])
X_extreme_norm = scaler.transform(X_extreme)

print("\n=== ADVERSARIAL : donnee hors distribution ===")
print(f"MedInc brut      : {X_extreme[0, 0]}")
print(f"MedInc normalise : {X_extreme_norm[0, 0]:.2f}")
# Valeur normalisee ~ +52000 : le reseau n'a rien vu d'aussi grand -> prediction non fiable
print(f"Toutes valeurs normalisees : {X_extreme_norm.round(1)}")
