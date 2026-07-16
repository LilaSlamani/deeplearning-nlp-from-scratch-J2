import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

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
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(8,),
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    if use_dropout:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    if use_dropout:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

# Happy path : l'écart loss_train - val_loss doit être plus petit avec L2 qu'en baseline
print("=== HAPPY PATH : L2 réduit l'overfit ===")
model_base = build_pima_regularized(l2_lambda=0.0)
model_l2   = build_pima_regularized(l2_lambda=0.01)
h_base = model_base.fit(X_train_norm, y_train, epochs=100, validation_split=0.2, verbose=0)
h_l2   = model_l2.fit(X_train_norm, y_train, epochs=100, validation_split=0.2, verbose=0)
# Valeur négative = val_loss > train_loss -> overfit. Plus c'est proche de 0, moins on overfit.
gap_base = h_base.history['loss'][-1] - h_base.history['val_loss'][-1]
gap_l2   = h_l2.history['loss'][-1]   - h_l2.history['val_loss'][-1]
print(f"Écart train-val baseline : {gap_base:.4f}")
print(f"Écart train-val L2       : {gap_l2:.4f}")

# Cas limite : en inférence Keras désactive le Dropout automatiquement (training=False implicite)
print("\n=== CAS LIMITE : Dropout désactivé en inférence ===")
model_drop = build_pima_regularized(l2_lambda=0.01, use_dropout=True)
model_drop.fit(X_train_norm, y_train, epochs=20, validation_split=0.2, verbose=0)
pred1 = model_drop.predict(X_test_norm[:5], verbose=0)
pred2 = model_drop.predict(X_test_norm[:5], verbose=0)
# Si Dropout était actif en inférence, pred1 et pred2 seraient différents à chaque appel
assert np.allclose(pred1, pred2), "Dropout actif en inférence - bug !"
print("OK - deux appels predict identiques (Dropout bien désactivé)")

# Adversarial : lambda=10 force les poids à 0 -> le modèle ne discrimine plus rien
print("\n=== ADVERSARIAL : L2 lambda=10 (trop fort) ===")
model_fort = build_pima_regularized(l2_lambda=10.0)
h_fort = model_fort.fit(X_train_norm, y_train, epochs=50, validation_split=0.2, verbose=0)
acc_fort = max(h_fort.history['val_accuracy'])
print(f"val_accuracy avec lambda=10 : {acc_fort:.4f} (attendu proche de 0.65 = underfit)")
