import os

# Happy path : vérifié dans TensorBoard - norm converge, raw oscille
print("=== HAPPY PATH ===")
print("california_norm : loss stable ~0.3 | california_raw : oscillations jusqu'à 5+")

# Cas limite : logdir inexistant -> TensorBoard démarre mais "No dashboards are active"
# Piège courant : on croit que l'entraînement a raté, c'est juste le chemin qui est faux
print("\n=== CAS LIMITE ===")
print("Commande : tensorboard --logdir=logs/fit_vide")
print("Attendu  : 'No dashboards are active' (pas d'erreur, juste rien à afficher)")

# Adversarial : deux instances sur le même port -> la seconde refuse de démarrer
# Réflexe production : tuer le premier process avant de relancer
print("\n=== ADVERSARIAL ===")
print("Lancer deux fois tensorboard sur port 6006 -> 'address already in use'")

# Vérification automatique : les dossiers de logs existent
log_base = "logs/fit"
if os.path.exists(log_base):
    runs = os.listdir(log_base)
    print(f"\n=== LOGS ===")
    print(f"norm : {[r for r in runs if 'california_norm' in r]}")
    print(f"raw  : {[r for r in runs if 'california_raw'  in r]}")
else:
    print(f"\nDossier {log_base} introuvable")
