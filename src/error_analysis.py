"""
Error Analysis : pourquoi le modèle rate presque toutes les attaques R2L ?

On analyse :
1. La répartition des sous-types précis d'attaques (pas juste la catégorie)
2. Leur présence dans le train vs test set
3. La confiance du modèle sur ses erreurs
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from load_data import load_nsl_kdd
from preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUT_DIR = os.path.join(BASE_DIR, "..", "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")

# ---------- 1. Charger les labels détaillés (avant regroupement en catégories) ----------
train_raw, test_raw = load_nsl_kdd(TRAIN_PATH, TEST_PATH)

print("=" * 70)
print("ANALYSE 1 : sous-types d'attaques R2L - présence train vs test")
print("=" * 70)

r2l_types = [
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
    "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
    "snmpguess", "xlock", "xsnoop", "worm", "httptunnel",
]

train_counts = train_raw["label"].value_counts()
test_counts = test_raw["label"].value_counts()

comparison_rows = []
for attack_type in r2l_types:
    n_train = train_counts.get(attack_type, 0)
    n_test = test_counts.get(attack_type, 0)
    comparison_rows.append({"attack_type": attack_type, "n_train": n_train, "n_test": n_test})

r2l_comparison = pd.DataFrame(comparison_rows)
r2l_comparison["seen_in_train"] = r2l_comparison["n_train"] > 0
r2l_comparison = r2l_comparison[r2l_comparison["n_test"] > 0].sort_values("n_test", ascending=False)

print(r2l_comparison.to_string(index=False))

n_unseen = (~r2l_comparison["seen_in_train"]).sum()
n_total_types = len(r2l_comparison)
n_unseen_examples = r2l_comparison.loc[~r2l_comparison["seen_in_train"], "n_test"].sum()
n_total_examples = r2l_comparison["n_test"].sum()

print(f"\n>>> {n_unseen}/{n_total_types} sous-types de R2L dans le test sont ABSENTS du train.")
print(f">>> Cela représente {n_unseen_examples}/{n_total_examples} exemples de test "
      f"({100*n_unseen_examples/n_total_examples:.1f}%) que le modèle n'a JAMAIS vus à l'entraînement.")

# ---------- 2. Graphique : train vs test par sous-type ----------
plt.figure(figsize=(10, 6))
x = np.arange(len(r2l_comparison))
width = 0.35

plt.bar(x - width/2, r2l_comparison["n_train"], width, label="Train", color="#3498db")
plt.bar(x + width/2, r2l_comparison["n_test"], width, label="Test", color="#e74c3c")
plt.xticks(x, r2l_comparison["attack_type"], rotation=45, ha="right")
plt.ylabel("Nombre d'exemples")
plt.title("Sous-types R2L : nombre d'exemples Train vs Test\n(barre bleue à 0 = attaque jamais vue à l'entraînement)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "11_r2l_train_vs_test.png"), dpi=120)
plt.close()
print(f"\n✔ Graphique sauvegardé dans notebooks/11_r2l_train_vs_test.png")

# ---------- 3. Confiance du modèle sur les erreurs R2L ----------
print("\n" + "=" * 70)
print("ANALYSE 2 : confiance du modèle quand il se trompe sur R2L")
print("=" * 70)

data = preprocess(TRAIN_PATH, TEST_PATH)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train_multi"], data["y_test_multi"]
class_names = list(data["category_encoder"].classes_)

model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.3,
    objective="multi:softmax", num_class=len(class_names),
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train)

# On récupère les probabilités (pas juste la classe prédite)
y_proba = model.predict_proba(X_test)
y_pred = np.argmax(y_proba, axis=1)

r2l_idx = class_names.index("r2l")
normal_idx = class_names.index("normal")

# Cas où la réalité est r2l mais la prédiction est normal (l'erreur qu'on veut comprendre)
mask_r2l_missed = (y_test.values == r2l_idx) & (y_pred == normal_idx)
proba_normal_on_missed = y_proba[mask_r2l_missed, normal_idx]
proba_r2l_on_missed = y_proba[mask_r2l_missed, r2l_idx]

print(f"Nombre de R2L classés à tort comme 'normal' : {mask_r2l_missed.sum()}")
print(f"Probabilité moyenne donnée à 'normal' sur ces erreurs : {proba_normal_on_missed.mean():.3f}")
print(f"Probabilité moyenne donnée à 'r2l' sur ces erreurs     : {proba_r2l_on_missed.mean():.3f}")

plt.figure(figsize=(8, 5))
plt.hist(proba_normal_on_missed, bins=20, color="#c0392b", alpha=0.8)
plt.axvline(proba_normal_on_missed.mean(), color="black", linestyle="--",
            label=f"Moyenne = {proba_normal_on_missed.mean():.2f}")
plt.xlabel("Probabilité donnée à la classe 'normal'")
plt.ylabel("Nombre de cas")
plt.title("Confiance du modèle quand il classe à tort une attaque R2L comme normale")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "12_confidence_r2l_errors.png"), dpi=120)
plt.close()
print(f"✔ Graphique sauvegardé dans notebooks/12_confidence_r2l_errors.png")

print("\n" + "=" * 70)
print("CONCLUSION DE L'ERROR ANALYSIS")
print("=" * 70)
print("""
Deux causes probables combinées expliquent le F1-score quasi nul sur R2L :

1. GENERALIZATION ZERO-DAY : une partie significative des sous-types R2L du
   test set n'existent pas du tout dans le train set. Le modèle ne peut pas
   apprendre à détecter une attaque qu'il n'a jamais vue.

2. SIMILARITE AVEC LE TRAFIC NORMAL : les attaques R2L (vol de mot de passe,
   accès FTP non autorisé...) se déroulent souvent via des connexions dont les
   caractéristiques RESEAU (durée, bytes, flags) ressemblent énormément à du
   trafic légitime. La différence se joue au niveau applicatif (contenu des
   paquets), pas dans les métadonnées utilisées comme features ici.
""")
