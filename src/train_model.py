"""
Entraînement d'un premier modèle Random Forest (baseline).
Classification BINAIRE : normal (0) vs attaque (1)
"""
import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
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
y_train, y_test = data["y_train_bin"], data["y_test_bin"]

print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")

# ---------- 2. Entraîner le modèle ----------
print("\nEntraînement du Random Forest...")
start = time.time()

model = RandomForestClassifier(
    n_estimators=100,      # nombre d'arbres
    max_depth=None,        # profondeur non limitée
    random_state=42,       # reproductibilité
    n_jobs=-1,              # utilise tous les coeurs CPU disponibles
)
model.fit(X_train, y_train)

elapsed = time.time() - start
print(f"Entraînement terminé en {elapsed:.1f} secondes.")

# ---------- 3. Prédictions ----------
y_pred = model.predict(X_test)

# ---------- 4. Évaluation ----------
print("\n" + "=" * 60)
print("RÉSULTATS - Classification binaire (normal vs attaque)")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy : {acc:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nRapport de classification détaillé :")
print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

# ---------- 5. Matrice de confusion ----------
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["normal", "attack"], yticklabels=["normal", "attack"],
)
plt.xlabel("Prédiction")
plt.ylabel("Réalité")
plt.title("Matrice de confusion - Random Forest (binaire)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_confusion_matrix_rf_binary.png"), dpi=120)
plt.close()
print(f"\n✔ Matrice de confusion sauvegardée dans notebooks/04_confusion_matrix_rf_binary.png")

# ---------- 6. Importance des features (bonus intéressant) ----------
importances = pd.Series(model.feature_importances_, index=data["feature_cols"])
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
top_features.sort_values().plot(kind="barh", color="#2980b9")
plt.title("Top 15 des features les plus importantes (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_feature_importance_rf.png"), dpi=120)
plt.close()
print(f"✔ Importance des features sauvegardée dans notebooks/05_feature_importance_rf.png")

print("\nTop 10 features les plus importantes :")
print(top_features.head(10))
