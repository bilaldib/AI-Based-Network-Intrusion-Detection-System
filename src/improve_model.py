"""
Amélioration du modèle multi-classe : comparaison de 3 approches
1. Baseline       : XGBoost sans rééquilibrage (référence, = train_multiclass.py)
2. Class weights  : on pénalise plus fort les erreurs sur les classes rares
3. SMOTE          : on génère des exemples synthétiques pour les classes minoritaires

Objectif : voir si on peut améliorer la détection de R2L et U2R.
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE

from preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUT_DIR = os.path.join(BASE_DIR, "..", "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)

print("Chargement et prétraitement des données...")
data = preprocess(
    os.path.join(DATA_DIR, "KDDTrain+.txt"),
    os.path.join(DATA_DIR, "KDDTest+.txt"),
)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train_multi"], data["y_test_multi"]
class_names = list(data["category_encoder"].classes_)

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.3,
    objective="multi:softmax", num_class=len(class_names),
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)


def run_experiment(name, X_tr, y_tr, sample_weight=None):
    print(f"\n{'='*60}\nEntraînement : {name}\n{'='*60}")
    start = time.time()
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"Temps : {elapsed:.1f}s | F1-macro : {f1_macro:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    return {
        "name": name,
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(class_names, f1_per_class)),
        "train_time": elapsed,
    }


results = []

# ---------- 1. Baseline (rappel) ----------
results.append(run_experiment("Baseline (sans rééquilibrage)", X_train, y_train))

# ---------- 2. Class weights ----------
# compute_sample_weight donne un poids plus fort aux classes sous-représentées
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
results.append(run_experiment("Class weights (balanced)", X_train, y_train, sample_weight=sample_weights))

# ---------- 3. SMOTE ----------
print(f"\n{'='*60}\nApplication de SMOTE (génération d'exemples synthétiques)...\n{'='*60}")
print("Répartition AVANT SMOTE :")
print(pd.Series(y_train).map(dict(enumerate(class_names))).value_counts())

# k_neighbors=1 car u2r n'a que 52 exemples -> SMOTE a besoin de moins de voisins que d'exemples
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nRépartition APRÈS SMOTE :")
print(pd.Series(y_train_smote).map(dict(enumerate(class_names))).value_counts())

results.append(run_experiment("SMOTE (oversampling)", X_train_smote, y_train_smote))

# ---------- 4. Tableau comparatif ----------
print(f"\n{'='*60}\nTABLEAU COMPARATIF FINAL\n{'='*60}")

comparison = pd.DataFrame([
    {"approach": r["name"], "f1_macro": r["f1_macro"], **r["f1_per_class"]}
    for r in results
]).set_index("approach").round(4)

print(comparison)
comparison.to_csv(os.path.join(OUT_DIR, "improvement_comparison.csv"))
print(f"\n✔ Tableau sauvegardé dans notebooks/improvement_comparison.csv")

# ---------- 5. Graphique : F1 par classe, 3 approches côte à côte ----------
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(class_names))
width = 0.25
colors = ["#3498db", "#e67e22", "#2ecc71"]

for i, r in enumerate(results):
    scores = [r["f1_per_class"][c] for c in class_names]
    ax.bar(x + (i - 1) * width, scores, width, label=r["name"], color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_ylabel("F1-score")
ax.set_title("F1-score par classe : Baseline vs Class Weights vs SMOTE")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "13_improvement_comparison.png"), dpi=120)
plt.close()
print(f"✔ Graphique sauvegardé dans notebooks/13_improvement_comparison.png")

# ---------- 6. Conclusion automatique ----------
print(f"\n{'='*60}\nCONCLUSION\n{'='*60}")
for target_class in ["r2l", "u2r"]:
    print(f"\nÉvolution du F1-score pour '{target_class}' :")
    for r in results:
        print(f"  {r['name']:35s} -> {r['f1_per_class'][target_class]:.4f}")

best_macro = max(results, key=lambda r: r["f1_macro"])
print(f"\nMeilleure approche globale (F1-macro) : {best_macro['name']} ({best_macro['f1_macro']:.4f})")
