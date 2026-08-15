"""
Classification multi-classe avec un MLP (Multi-Layer Perceptron) - TensorFlow/Keras.
Objectif : comparer le Deep Learning avec les modèles classiques (Random Forest, XGBoost)
sur la même tâche : identifier le type d'attaque (normal/dos/probe/r2l/u2r).
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # masque les logs verbeux de TensorFlow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from preprocess import preprocess

# Reproductibilité
tf.random.set_seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUT_DIR = os.path.join(BASE_DIR, "..", "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1. Charger les données prétraitées ----------
print("Chargement et prétraitement des données...")
data = preprocess(
    os.path.join(DATA_DIR, "KDDTrain+.txt"),
    os.path.join(DATA_DIR, "KDDTest+.txt"),
)

X_train, X_test = data["X_train"].values, data["X_test"].values
y_train, y_test = data["y_train_multi"].values, data["y_test_multi"].values
class_names = list(data["category_encoder"].classes_)
n_classes = len(class_names)
n_features = X_train.shape[1]

print(f"X_train : {X_train.shape} | X_test : {X_test.shape} | Classes : {class_names}")

# ---------- 2. Poids de classes (même logique que class_weight="balanced" côté sklearn) ----------
class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights_array))
print(f"\nPoids par classe (pour compenser le déséquilibre) : "
      f"{dict(zip(class_names, np.round(class_weights_array, 2)))}")

# ---------- 3. Construction du modèle MLP ----------
model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(n_classes, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nArchitecture du MLP :")
model.summary()

# ---------- 4. Entraînement ----------
print("\nEntraînement du MLP...")
start = time.time()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=256,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=2,
)

elapsed = time.time() - start
n_epochs_run = len(history.history["loss"])
print(f"\nEntraînement terminé en {elapsed:.1f}s ({n_epochs_run} epochs, arrêt anticipé si applicable).")

# ---------- 5. Évaluation ----------
y_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_proba, axis=1)

f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("\n" + "=" * 60)
print("RÉSULTATS - MLP (TensorFlow/Keras)")
print("=" * 60)
print(f"F1-score macro    : {f1_macro:.4f}")
print(f"F1-score weighted : {f1_weighted:.4f}")
print("\nRapport de classification détaillé :")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# ---------- 6. Courbes d'entraînement (loss/accuracy) ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history.history["loss"], label="Train")
axes[0].plot(history.history["val_loss"], label="Validation")
axes[0].set_title("Évolution de la loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train")
axes[1].plot(history.history["val_accuracy"], label="Validation")
axes[1].set_title("Évolution de l'accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "14_mlp_training_curves.png"), dpi=120)
plt.close()
print(f"\n✔ Courbes d'entraînement sauvegardées dans notebooks/14_mlp_training_curves.png")

# ---------- 7. Matrice de confusion ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prédiction")
plt.ylabel("Réalité")
plt.title("Matrice de confusion - MLP (TensorFlow/Keras)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "15_confusion_matrix_mlp.png"), dpi=120)
plt.close()
print(f"✔ Matrice de confusion sauvegardée dans notebooks/15_confusion_matrix_mlp.png")

# ---------- 8. Comparaison finale avec les modèles classiques ----------
comp_path = os.path.join(OUT_DIR, "improvement_comparison.csv")
per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
mlp_row = {"approach": "MLP (TensorFlow, class_weight)", "f1_macro": f1_macro}
mlp_row.update(dict(zip(class_names, per_class_f1)))

if os.path.exists(comp_path):
    existing = pd.read_csv(comp_path, index_col=0)
    combined = pd.concat([existing, pd.DataFrame([mlp_row]).set_index("approach")])
    combined = combined.round(4)
    combined.to_csv(os.path.join(OUT_DIR, "final_model_comparison.csv"))
    print(f"\n✔ Comparaison finale (tous modèles) sauvegardée dans notebooks/final_model_comparison.csv")
    print("\n" + "=" * 60)
    print("TABLEAU COMPARATIF FINAL - TOUS LES MODÈLES")
    print("=" * 60)
    print(combined)

    # Graphique final tous modèles confondus
    plt.figure(figsize=(11, 6))
    combined["f1_macro"].sort_values().plot(kind="barh", color="#16a085")
    plt.xlabel("F1-score macro")
    plt.title("Comparaison finale de tous les modèles testés")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "16_final_comparison_all_models.png"), dpi=120)
    plt.close()
    print(f"✔ Graphique final sauvegardé dans notebooks/16_final_comparison_all_models.png")
else:
    print("\n(Lance d'abord improve_model.py pour avoir une comparaison complète avec les autres modèles.)")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print(f"""
Le MLP obtient un F1-macro de {f1_macro:.4f}.
Comparaison attendue : XGBoost+SMOTE atteint généralement un F1-macro proche
ou supérieur sur ce dataset tabulaire structuré. C'est cohérent avec la
littérature : les modèles à base d'arbres (Random Forest, XGBoost) dominent
généralement le Deep Learning classique sur des données tabulaires, le Deep
Learning excellant surtout sur des données non-structurées (images, texte,
séquences temporelles brutes).
""")
