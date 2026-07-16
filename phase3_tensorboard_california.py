import datetime
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Pipeline de référence (identique phase 1 et 2)
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm   = scaler.transform(X_val)
X_test_norm  = scaler.transform(X_test)

def build_regression_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_with_tensorboard(X_train, y_train, X_val, y_val, run_name, epochs=100):
    """Entraîne un modèle de régression avec un callback TensorBoard horodaté."""

    # Dossier unique par run : le timestamp évite d'écraser les logs précédents
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = f"logs/fit/{run_name}_" + timestamp

    # Le callback écrit loss/mae/val_loss/val_mae + histogrammes des poids à chaque epoch
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = build_regression_model(input_dim=8)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[tb_callback],
        verbose=0
    )

    print(f"Run '{run_name}' terminé. Logs dans {log_dir}")
    return model, history


# Run 1 : données normalisées (convergence attendue propre)
model_norm, history_norm = train_with_tensorboard(
    X_train_norm, y_train, X_val_norm, y_val,
    run_name="california_norm"
)

# Run 2 : données brutes (comportement dégradé à observer dans TensorBoard)
model_raw, history_raw = train_with_tensorboard(
    X_train, y_train, X_val, y_val,
    run_name="california_raw"
)

# Lancer TensorBoard dans un terminal séparé :
# tensorboard --logdir=logs/fit
# Puis ouvrir http://localhost:6006
