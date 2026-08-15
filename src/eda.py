"""
Analyse exploratoire (EDA) du dataset NSL-KDD.
Génère des graphiques pour comprendre la distribution des classes et des features.
"""
import os
import matplotlib
matplotlib.use("Agg")  # backend sans interface graphique
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from load_data import load_nsl_kdd

sns.set_theme(style="whitegrid")

# Chemins relatifs : src/ -> ../data/ et ../notebooks/ (fonctionne sur n'importe quel PC)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUT_DIR = os.path.join(BASE_DIR, "..", "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)

train, test = load_nsl_kdd(
    os.path.join(DATA_DIR, "KDDTrain+.txt"),
    os.path.join(DATA_DIR, "KDDTest+.txt"),
)

print("=" * 60)
print("APERÇU GÉNÉRAL")
print("=" * 60)
print(f"Nombre de lignes (train) : {train.shape[0]}")
print(f"Nombre de colonnes       : {train.shape[1]}")
print(f"Valeurs manquantes       : {train.isnull().sum().sum()}")
print(f"Doublons                 : {train.duplicated().sum()}")

print("\n" + "=" * 60)
print("RÉPARTITION normal vs attaque")
print("=" * 60)
binary = train["label"].apply(lambda x: "normal" if x == "normal" else "attack")
print(binary.value_counts())
print(binary.value_counts(normalize=True).round(3) * 100, "%")

print("\n" + "=" * 60)
print("RÉPARTITION PAR CATÉGORIE D'ATTAQUE (4 grandes classes NSL-KDD)")
print("=" * 60)
print(train["attack_category"].value_counts())

print("\n" + "=" * 60)
print("TOP 10 DES LABELS DÉTAILLÉS (types d'attaques précis)")
print("=" * 60)
print(train["label"].value_counts().head(10))

# ---------- Graphique 1 : répartition normal vs attaque ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

binary.value_counts().plot(
    kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"]
)
axes[0].set_title("Trafic normal vs attaque (binaire)")
axes[0].set_xlabel("")
axes[0].set_ylabel("Nombre de connexions")
axes[0].tick_params(axis="x", rotation=0)

train["attack_category"].value_counts().plot(
    kind="bar", ax=axes[1], color="#3498db"
)
axes[1].set_title("Répartition par catégorie (normal / dos / probe / r2l / u2r)")
axes[1].set_xlabel("")
axes[1].set_ylabel("Nombre de connexions")
axes[1].set_yscale("log")  # échelle log car très déséquilibré (u2r très rare)
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=120)
plt.close()
print(f"\n✔ Graphique sauvegardé : 01_class_distribution.png")

# ---------- Graphique 2 : protocole et service ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train["protocol_type"].value_counts().plot(
    kind="bar", ax=axes[0], color="#9b59b6"
)
axes[0].set_title("Répartition par protocole")
axes[0].tick_params(axis="x", rotation=0)

train["service"].value_counts().head(10).plot(
    kind="bar", ax=axes[1], color="#f39c12"
)
axes[1].set_title("Top 10 des services réseau")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_protocol_service.png", dpi=120)
plt.close()
print(f"✔ Graphique sauvegardé : 02_protocol_service.png")

# ---------- Graphique 3 : matrice de corrélation (features numériques) ----------
numeric_cols = train.select_dtypes(include=["int64", "float64"]).columns
numeric_cols = [c for c in numeric_cols if c != "difficulty"]
corr = train[numeric_cols].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.7})
plt.title("Matrice de corrélation entre les features numériques")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_correlation_matrix.png", dpi=120)
plt.close()
print(f"✔ Graphique sauvegardé : 03_correlation_matrix.png")

print("\n" + "=" * 60)
print("EDA terminée.")
print("=" * 60)
