import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df_wine = pd.read_csv(wine_url, sep=';')


def map_quality(q):
    if q <= 4: return 0
    elif q <= 6: return 1
    else: return 2


df_wine['quality_3class'] = df_wine['quality'].apply(map_quality)
X_wine = df_wine.drop(['quality', 'quality_3class'], axis=1).values
y_wine = df_wine['quality_3class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine
)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)


def build_wine_model():
    m = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(11,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


# Happy path : softmax -> chaque ligne de prédiction somme à 1
print("=== HAPPY PATH : softmax somme à 1 ===")
model = build_wine_model()
model.fit(X_train_norm, y_train, epochs=5, validation_split=0.2, verbose=0)
preds = model.predict(X_test_norm[:5], verbose=0)
for i, p in enumerate(preds):
    assert abs(p.sum() - 1.0) < 1e-5, f"Exemple {i} : somme={p.sum():.6f} != 1"
    print(f"Exemple {i} : {p.round(3)} -> somme={p.sum():.4f}")

# Cas limite : sans stratify la classe 0 (~4%) peut disparaître du train
print("\n=== CAS LIMITE : stratify obligatoire ===")
_, _, y_no_strat, _ = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)
_, _, y_strat,    _ = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine)
print(f"Sans stratify - classe 0 dans train : {(y_no_strat == 0).sum()} exemples")
print(f"Avec stratify - classe 0 dans train : {(y_strat == 0).sum()} exemples")

# Adversarial : sigmoid à la sortie au lieu de softmax -> somme != 1
print("\n=== ADVERSARIAL : sigmoid au lieu de softmax ===")
model_bad = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(11,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='sigmoid')  # bug : 3 sorties indépendantes, pas normalisées
])
model_bad.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_bad.fit(X_train_norm, y_train, epochs=5, validation_split=0.2, verbose=0)
preds_bad = model_bad.predict(X_test_norm[:3], verbose=0)
for i, p in enumerate(preds_bad):
    print(f"Exemple {i} : {p.round(3)} -> somme={p.sum():.4f} (doit != 1)")
