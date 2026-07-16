import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)

# Distribution : 500 non-diabétiques vs 268 diabétiques (classe déséquilibrée)
print("Distribution Outcome :")
print(df['Outcome'].value_counts())

# Zéros dans Glucose/BMI/Insulin = NaN déguisés (physiologiquement impossibles)
print("\nZéros par colonne :")
print((df == 0).sum())

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)

# Classification binaire : sigmoid donne P(diabète) entre 0 et 1
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(8,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    verbose=1
)

# Max sur toutes les epochs : la dernière epoch n'est pas forcément la meilleure
print(f"\nMeilleure val_accuracy : {max(history.history['val_accuracy']):.4f}")

# Sanity check : moyenne ~0.35 sinon le modèle prédit toujours 0 (classe majoritaire)
preds_mean = model.predict(X_test_norm).mean()
print(f"Moyenne des prédictions : {preds_mean:.4f} (attendu ~0.35)")
