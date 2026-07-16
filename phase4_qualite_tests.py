import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

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


def build_model():
    m = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


# Happy path : dépasser le seuil de la classe majoritaire (65% en prédisant toujours 0)
print("=== HAPPY PATH ===")
model = build_model()
history = model.fit(X_train_norm, y_train, epochs=30, validation_split=0.2, verbose=0)
best_val_acc = max(history.history['val_accuracy'])
assert best_val_acc > 0.65, f"Pas meilleur que la classe majoritaire : {best_val_acc:.4f}"
print(f"OK - val_accuracy max : {best_val_acc:.4f} > 0.65")

# Cas limite : sigmoid reste dans [0,1] même avec des entrées extrêmes
print("\n=== CAS LIMITE : entrée hors distribution ===")
extreme_input = np.array([[99999, -99999, 0, 0, 0, 0, 0, 0]])
pred_extreme  = model.predict(scaler.transform(extreme_input), verbose=0)[0][0]
assert 0.0 <= pred_extreme <= 1.0, "sigmoid hors [0,1] !"
print(f"OK - prédiction extrême : {pred_extreme:.4f} (dans [0,1])")

# Adversarial : relu à la sortie au lieu de sigmoid -> collapse des gradients négatifs
print("\n=== ADVERSARIAL : relu à la sortie ===")
model_bad = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(8,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='relu')  # bug : relu bloque les gradients négatifs
])
model_bad.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_bad = model_bad.fit(X_train_norm, y_train, epochs=10, validation_split=0.2, verbose=0)
acc_bad = max(history_bad.history['val_accuracy'])
print(f"relu final  - val_accuracy : {acc_bad:.4f}")
print(f"sigmoid final - val_accuracy : {best_val_acc:.4f}")
