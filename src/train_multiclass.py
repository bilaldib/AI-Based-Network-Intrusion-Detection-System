"""
Classification MULTI-CLASSE avec XGBoost.
Objectif : identifier le TYPE d'attaque, pas juste normal/attaque.
Classes : normal / dos / probe / r2l / u2r
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from preprocess import preprocess

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

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train_multi"], data["y_test_multi"]
class_names = list(data["category_encoder"].classes_)  # ex: ['dos','normal','probe','r2l','u2r']

print(f"Classes : {class_names}")
print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")

# ---------- 2. Entraînement XGBoost multi-classe ----------
print("\nEntraînement XGBoost (multi-classe)...")
start = time.time()

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.3,
    objective="multi:softmax",
    num_class=len(class_names),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

elapsed = time.time() - start
print(f"Entraînement terminé en {elapsed:.1f} secondes.")

# ---------- 3. Prédictions et évaluation ----------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("\n" + "=" * 60)
print("RÉSULTATS - Classification multi-classe (XGBoost)")
print("=" * 60)
print(f"Accuracy      : {acc:.4f}")
print(f"F1-score macro    : {f1_macro:.4f}  (moyenne simple entre classes, ne favorise pas les classes majoritaires)")
print(f"F1-score weighted : {f1_weighted:.4f}  (pondéré par le nombre d'exemples par classe)")

print("\nRapport de classification détaillé (par classe) :")
report = classification_report(
    y_test, y_pred, target_names=class_names, zero_division=0
)
print(report)

# ---------- 4. Matrice de confusion 5x5 ----------
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion (lignes = réalité, colonnes = prédiction) :")
print(pd.DataFrame(cm, index=class_names, columns=class_names))

plt.figure(figsize=(8, 7))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Purples",
    xticklabels=class_names, yticklabels=class_names,
)
plt.xlabel("Prédiction")
plt.ylabel("Réalité")
plt.title("Matrice de confusion - XGBoost (multi-classe)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_confusion_matrix_multiclass.png"), dpi=120)
plt.close()
print(f"\n✔ Matrice de confusion sauvegardée dans notebooks/08_confusion_matrix_multiclass.png")

# ---------- 5. Version normalisée (en %) - plus lisible pour les classes rares ----------
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
cm_normalized = np.nan_to_num(cm_normalized)  # gère la division par zéro si une classe est absente

plt.figure(figsize=(8, 7))
sns.heatmap(
    cm_normalized, annot=True, fmt=".2f", cmap="Purples",
    xticklabels=class_names, yticklabels=class_names,
    vmin=0, vmax=1,
)
plt.xlabel("Prédiction")
plt.ylabel("Réalité")
plt.title("Matrice de confusion normalisée (% par classe réelle)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_confusion_matrix_multiclass_normalized.png"), dpi=120)
plt.close()
print(f"✔ Matrice normalisée sauvegardée dans notebooks/09_confusion_matrix_multiclass_normalized.png")

# ---------- 6. F1-score par classe (graphique) ----------
per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
support = pd.Series(y_test).value_counts().sort_index()

f1_df = pd.DataFrame({
    "class": class_names,
    "f1_score": per_class_f1,
    "support_test": support.values,
})
f1_df = f1_df.sort_values("f1_score")

fig, ax1 = plt.subplots(figsize=(9, 5))
bars = ax1.barh(f1_df["class"], f1_df["f1_score"], color="#8e44ad")
ax1.set_xlabel("F1-score")
ax1.set_xlim(0, 1)
ax1.set_title("F1-score par classe (avec nombre d'exemples de test)")

for bar, support_val in zip(bars, f1_df["support_test"]):
    ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
              f"n={support_val}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "10_f1_per_class.png"), dpi=120)
plt.close()
print(f"✔ F1-score par classe sauvegardé dans notebooks/10_f1_per_class.png")

print("\n" + "=" * 60)
print("OBSERVATIONS CLÉS")
print("=" * 60)
print(f1_df.to_string(index=False))
