import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Preprocessing identique à la phase 1 (pipeline de référence)
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm   = scaler.transform(X_val)
X_test_norm  = scaler.transform(X_test)

def build_regression_model(input_dim):
    # Pas d'activation sur la sortie : avec sigmoid la sortie serait bloquée
    # entre 0 et 1, or les prix vont jusqu'à 5 (500k$). Bug silencieux.
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # régression : valeur continue, pas d'activation
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_regression_model(input_dim=8)
model.summary()

history = model.fit(X_train_norm, y_train, epochs=100, batch_size=32, validation_data=(X_val_norm, y_val), verbose=1)

test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"MAE test : {test_mae:.4f} (en centaines de milliers de $)")