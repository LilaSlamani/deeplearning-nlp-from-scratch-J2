import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Pipeline de référence
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm   = scaler.transform(X_val)
X_test_norm  = scaler.transform(X_test)

def build_regression_model(input_dim):
    # Pas d'activation sur la sortie : régression = valeur continue libre
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# === CAS LIMITE : batch_size=1 (SGD pur) vs batch_size=total (Batch GD) ===
# batch_size=1 : 13 209 mises à jour par epoch, très bruité
# batch_size=total : 1 seule mise à jour par epoch, très lisse mais lent

print("=== CAS LIMITE : impact du batch_size ===")

# SGD pur : un exemple à la fois -> gradient très bruité
model_sgd = build_regression_model(8)
history_sgd = model_sgd.fit(
    X_train_norm, y_train,
    epochs=10, batch_size=1,
    validation_data=(X_val_norm, y_val),
    verbose=0
)
val_loss_sgd = history_sgd.history['val_loss'][-1]
print(f"batch_size=1 (SGD pur)   -> val_loss epoch 10 : {val_loss_sgd:.4f}")

# Batch GD : tous les exemples d'un coup -> 1 mise à jour par epoch
model_bgd = build_regression_model(8)
history_bgd = model_bgd.fit(
    X_train_norm, y_train,
    epochs=10, batch_size=len(X_train_norm),
    validation_data=(X_val_norm, y_val),
    verbose=0
)
val_loss_bgd = history_bgd.history['val_loss'][-1]
print(f"batch_size=total (BGD)   -> val_loss epoch 10 : {val_loss_bgd:.4f}")

# Lequel converge le mieux en 10 epochs ?
meilleur = "SGD pur (batch=1)" if val_loss_sgd < val_loss_bgd else "Batch GD (batch=total)"
print(f"Meilleur en 10 epochs : {meilleur}")


# === ADVERSARIAL : entraînement sans normalisation ===
# Latitude/Longitude ont des valeurs ~37-122, AveRooms ~5
# -> gradients déséquilibrés -> loss initiale très élevée

print("\n=== ADVERSARIAL : sans normalisation ===")

model_raw = build_regression_model(8)
history_raw = model_raw.fit(
    X_train, y_train,   # données brutes, non normalisées
    epochs=10, batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

val_loss_epoch1  = history_raw.history['val_loss'][0]
val_loss_epoch10 = history_raw.history['val_loss'][-1]
print(f"\nSans normalisation - val_loss epoch 1  : {val_loss_epoch1:.2f}")
print(f"Sans normalisation - val_loss epoch 10 : {val_loss_epoch10:.2f}")
print(f"Run normalisé      - val_loss epoch 10 : ~0.28 (référence phase 2)")
# Oscillations attendues : les grandes valeurs de Latitude/Longitude
# déstabilisent Adam -> la loss remonte brutalement certains epochs
