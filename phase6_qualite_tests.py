import os
import shutil
import keras_tuner as kt
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


def build_pima_model(hp):
    model = keras.Sequential()
    units_1       = hp.Int('units_1', min_value=32, max_value=128, step=32)
    units_2       = hp.Int('units_2', min_value=16, max_value=64,  step=16)
    activation    = hp.Choice('activation', values=['relu', 'tanh'])
    dropout_rate  = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    model.add(layers.Dense(units_1, activation=activation, input_shape=(8,)))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(units_2, activation=activation))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Happy path : deux tuners avec seed=42 doivent tirer exactement les mêmes configs aléatoires
print("=== HAPPY PATH : reproductibilité seed=42 ===")
early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner1 = kt.RandomSearch(build_pima_model, objective='val_accuracy',
                          max_trials=3, seed=42,
                          directory='tuning_test', project_name='test1')
tuner2 = kt.RandomSearch(build_pima_model, objective='val_accuracy',
                          max_trials=3, seed=42,
                          directory='tuning_test', project_name='test2')
tuner1.search(X_train_norm, y_train, epochs=10, validation_split=0.2, callbacks=[early], verbose=0)
tuner2.search(X_train_norm, y_train, epochs=10, validation_split=0.2, callbacks=[early], verbose=0)
hp1 = tuner1.get_best_hyperparameters()[0].values
hp2 = tuner2.get_best_hyperparameters()[0].values
print(f"Tuner 1 : {hp1}")
print(f"Tuner 2 : {hp2}")
# Si False ici, le seed ne fonctionne pas -> résultats non reproductibles entre machines
print(f"Identiques : {hp1 == hp2}")

# Cas limite : si tuning_pima/pima_random existe déjà, keras-tuner recharge sans réentraîner
# Comportement attendu : "Reloading Tuner from ..." dans le terminal au lieu de relancer tous les trials
print("\n=== CAS LIMITE : reprise depuis dossier existant ===")
print("Si tuning_pima/pima_random existe : keras-tuner recharge les trials déjà faits")
print("Attendu : message 'Reloading Tuner from ...' (pas de réentraînement inutile)")

# Adversarial : min_value == max_value -> une seule valeur possible, le tuner ne cherche rien
print("\n=== ADVERSARIAL : espace de recherche minimal ===")
def build_minimal(hp):
    m = keras.Sequential([
        # min=max=32 : le tuner n'a qu'un seul choix, tous les trials seront identiques
        layers.Dense(hp.Int('u', min_value=32, max_value=32, step=32),
                     activation='relu', input_shape=(8,)),
        layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

tuner_min = kt.RandomSearch(build_minimal, objective='val_accuracy',
                             max_trials=3, seed=42,
                             directory='tuning_test', project_name='minimal')
tuner_min.search(X_train_norm, y_train, epochs=5, validation_split=0.2, verbose=0)
print(f"Tous les trials ont u=32 : {tuner_min.get_best_hyperparameters()[0].values}")

# Nettoyage du dossier temporaire créé pour ces tests
if os.path.exists('tuning_test'):
    shutil.rmtree('tuning_test')
    print("\nDossier tuning_test supprimé")
