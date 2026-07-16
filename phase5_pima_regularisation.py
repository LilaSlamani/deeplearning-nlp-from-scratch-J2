import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)


def build_pima_regularized(l2_lambda=0.01, use_dropout=False):
    """Construit un modèle Pima avec L2 et Dropout optionnels selon les paramètres."""
    model = keras.Sequential()

    # L2 pénalise les grands poids : loss totale = binary_crossentropy + lambda * somme(poids²)
    model.add(layers.Dense(64, activation='relu', input_shape=(8,),
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    if use_dropout:
        # Dropout désactive 30% des neurones aléatoirement à chaque batch d'entraînement
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(32, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    if use_dropout:
        model.add(layers.Dropout(0.3))

    # Pas de régularisation sur la sortie : on ne veut pas contraindre les probabilités
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# restore_best_weights=True : récupère les poids de l'epoch avec la meilleure val_loss, pas du dernier
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

# Config 1 : aucune régularisation, sert de référence pour mesurer l'overfit de base
print("=== Config 1 : baseline ===")
model_baseline = build_pima_regularized(l2_lambda=0.0, use_dropout=False)
history_baseline = model_baseline.fit(
    X_train_norm, y_train, epochs=300,
    validation_split=0.2, callbacks=[early_stopping], verbose=0
)
epochs_baseline = len(history_baseline.history['val_loss'])
acc_baseline    = max(history_baseline.history['val_accuracy'])
print(f"Arrêt epoch {epochs_baseline} | val_accuracy max : {acc_baseline:.4f}")

# Config 2 : L2 seul ralentit la descente de loss train -> les courbes restent plus proches
print("=== Config 2 : L2 seul ===")
model_l2 = build_pima_regularized(l2_lambda=0.01, use_dropout=False)
history_l2 = model_l2.fit(
    X_train_norm, y_train, epochs=300,
    validation_split=0.2, callbacks=[early_stopping], verbose=0
)
epochs_l2 = len(history_l2.history['val_loss'])
acc_l2    = max(history_l2.history['val_accuracy'])
print(f"Arrêt epoch {epochs_l2} | val_accuracy max : {acc_l2:.4f}")

# Config 3 : L2 + Dropout combinés -> courbes train/val quasi confondues, meilleure généralisation
print("=== Config 3 : L2 + Dropout ===")
model_l2_drop = build_pima_regularized(l2_lambda=0.01, use_dropout=True)
history_l2_drop = model_l2_drop.fit(
    X_train_norm, y_train, epochs=300,
    validation_split=0.2, callbacks=[early_stopping], verbose=0
)
epochs_l2_drop = len(history_l2_drop.history['val_loss'])
acc_l2_drop    = max(history_l2_drop.history['val_accuracy'])
print(f"Arrêt epoch {epochs_l2_drop} | val_accuracy max : {acc_l2_drop:.4f}")

# Graphe comparatif : train vs val loss pour chaque config, ligne rouge = epoch d'arrêt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, history, title, n_ep in [
    (axes[0], history_baseline, "Baseline",     epochs_baseline),
    (axes[1], history_l2,       "L2 seul",      epochs_l2),
    (axes[2], history_l2_drop,  "L2 + Dropout", epochs_l2_drop),
]:
    ax.plot(history.history['loss'],     label='train')
    ax.plot(history.history['val_loss'], label='val')
    ax.axvline(x=n_ep - 1, color='red', linestyle='--', label=f'arrêt ({n_ep})')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

plt.tight_layout()
plt.savefig('phase5_pima_3configs.png')
print("\nGraphe sauvegardé : phase5_pima_3configs.png")
