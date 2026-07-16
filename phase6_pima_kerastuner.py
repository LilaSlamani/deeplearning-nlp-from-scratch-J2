import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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


def build_pima_model(hp):
    """Construit un modèle Pima dont les hyperparamètres sont décidés par keras-tuner à chaque trial."""
    model = keras.Sequential()

    # hp.Int/Choice/Float déclarent des plages : keras-tuner choisit une valeur dans chaque plage
    units_1       = hp.Int('units_1', min_value=32, max_value=128, step=32)
    units_2       = hp.Int('units_2', min_value=16, max_value=64,  step=16)
    activation    = hp.Choice('activation', values=['relu', 'tanh'])
    dropout_rate  = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

    model.add(layers.Dense(units_1, activation=activation, input_shape=(8,)))
    # Dropout optionnel : si le tuner choisit 0.0, on ne l'ajoute pas du tout
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(units_2, activation=activation))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    # Sortie binaire : sigmoid donne P(diabète)
    model.add(layers.Dense(1, activation='sigmoid'))

    # Adam avec lr variable : c'est un des hyperparamètres les plus impactants
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# seed=42 : les 15 configs tirées au hasard seront toujours les mêmes d'une exécution à l'autre
tuner = kt.RandomSearch(
    build_pima_model,
    objective='val_accuracy',   # on cherche à maximiser val_accuracy, pas val_loss
    max_trials=15,
    seed=42,
    directory='tuning_pima',    # les résultats sont sauvegardés ici, reprise possible si crash
    project_name='pima_random'
)

tuner.search_space_summary()

# patience=10 pendant la recherche : on coupe vite les mauvais trials pour ne pas perdre de temps
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

tuner.search(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# Affiche les 5 meilleurs trials pour identifier les invariants (hp qui reviennent toujours)
tuner.results_summary(num_trials=5)

print("\n=== INVARIANTS DANS LES 5 MEILLEURS TRIALS ===")
for i, hp in enumerate(tuner.get_best_hyperparameters(num_trials=5)):
    print(f"Trial {i+1} : {hp.values}")

best_hp = tuner.get_best_hyperparameters()[0]
print("\nMeilleur lr :", best_hp.get('learning_rate'))
print("Meilleures units_1 :", best_hp.get('units_1'))
print("Meilleures units_2 :", best_hp.get('units_2'))
print("Meilleure activation :", best_hp.get('activation'))
print("Meilleur dropout_rate :", best_hp.get('dropout_rate'))

# Réentraîner le meilleur modèle plus longtemps : pendant la recherche chaque trial était coupé tôt
best_model = tuner.hypermodel.build(best_hp)
history_best = best_model.fit(
    X_train_norm, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)
print(f"\nBest model val_accuracy : {max(history_best.history['val_accuracy']):.4f}")
