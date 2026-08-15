"""
Entraînement d'un modèle XGBoost (classification binaire) et comparaison avec Random Forest.
"""
import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
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


def evaluate_model(name, model, X_test, y_test, train_time):
    """Évalue un modèle et retourne un dictionnaire de métriques."""
    y_pred = model.predict(X_test)

    metrics = {
        "model": name,
        "train_time_s": train_time,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print(f"\n{'='*60}")
    print(f"RÉSULTATS - {name}")
    print(f"{'='*60}")
    print(f"Temps d'entraînement : {train_time:.1f}s")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :")
    print(cm)

    return metrics, cm, y_pred


# ---------- 2. Random Forest (rappel, pour comparaison directe) ----------
print("\nEntraînement Random Forest...")
start = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start

rf_metrics, rf_cm, rf_pred = evaluate_model("Random Forest", rf_model, X_test, y_test, rf_time)

# ---------- 3. XGBoost ----------
print("\nEntraînement XGBoost...")
start = time.time()
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start

xgb_metrics, xgb_cm, xgb_pred = evaluate_model("XGBoost", xgb_model, X_test, y_test, xgb_time)

# ---------- 4. Tableau comparatif ----------
comparison = pd.DataFrame([rf_metrics, xgb_metrics]).set_index("model")
comparison = comparison.round(4)

print("\n" + "=" * 60)
print("TABLEAU COMPARATIF")
print("=" * 60)
print(comparison)

comparison.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))
print(f"\n✔ Tableau sauvegardé dans notebooks/model_comparison.csv")

# ---------- 5. Graphique comparatif ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
comparison[metrics_to_plot].T.plot(kind="bar", ax=axes[0], color=["#3498db", "#e67e22"])
axes[0].set_title("Comparaison des métriques")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)
axes[0].legend(title="Modèle")
axes[0].tick_params(axis="x", rotation=30)

comparison["train_time_s"].plot(kind="bar", ax=axes[1], color=["#3498db", "#e67e22"])
axes[1].set_title("Temps d'entraînement (secondes)")
axes[1].set_ylabel("Secondes")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_rf_vs_xgboost.png"), dpi=120)
plt.close()
print(f"✔ Graphique comparatif sauvegardé dans notebooks/06_rf_vs_xgboost.png")

# ---------- 6. Matrices de confusion côte à côte ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["normal", "attack"], yticklabels=["normal", "attack"])
axes[0].set_title("Random Forest")
axes[0].set_xlabel("Prédiction")
axes[0].set_ylabel("Réalité")

sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Oranges", ax=axes[1],
            xticklabels=["normal", "attack"], yticklabels=["normal", "attack"])
axes[1].set_title("XGBoost")
axes[1].set_xlabel("Prédiction")
axes[1].set_ylabel("Réalité")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_confusion_matrices_comparison.png"), dpi=120)
plt.close()
print(f"✔ Matrices de confusion comparées sauvegardées dans notebooks/07_confusion_matrices_comparison.png")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
best_model = comparison["f1_score"].idxmax()
print(f"Meilleur modèle (selon F1-score) : {best_model}")
