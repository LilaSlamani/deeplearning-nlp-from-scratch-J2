import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df_wine = pd.read_csv(wine_url, sep=';')

print("Distribution des qualités brutes :")
print(df_wine['quality'].value_counts().sort_index())

# Agrégation en 3 classes : qualité brute va de 3 à 8, très déséquilibrée
def map_quality(q):
    if q <= 4:
        return 0  # low
    elif q <= 6:
        return 1  # medium
    else:
        return 2  # high

df_wine['quality_3class'] = df_wine['quality'].apply(map_quality)

print("\nDistribution agrégée (3 classes) :")
print(df_wine['quality_3class'].value_counts().sort_index())

X_wine = df_wine.drop(['quality', 'quality_3class'], axis=1).values
y_wine = df_wine['quality_3class'].values

# stratify indispensable : sans lui la classe 0 (~4%) peut ne pas apparaître dans train
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine
)

scaler = StandardScaler()
X_wine_train_norm = scaler.fit_transform(X_wine_train)
X_wine_test_norm  = scaler.transform(X_wine_test)

# 3 neurones de sortie avec softmax -> probabilités qui somment à 1
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(11,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# sparse_categorical : labels entiers (0, 1, 2) sans conversion en one-hot
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_wine_train_norm, y_wine_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_wine_test_norm, y_wine_test, verbose=0)
print(f"\nVal_accuracy finale : {history.history['val_accuracy'][-1]:.4f}")
print(f"Test accuracy : {test_acc:.4f}")
