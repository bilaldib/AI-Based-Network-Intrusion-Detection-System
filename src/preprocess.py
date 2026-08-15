"""
Prétraitement du dataset NSL-KDD :
- Encodage des variables catégorielles (protocol_type, service, flag)
- Préparation des labels (binaire + multi-classe)
- Normalisation des features numériques
- Split X / y prêt pour l'entraînement
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from load_data import load_nsl_kdd

# Chemins relatifs : src/ -> ../data/ (fonctionne sur n'importe quel PC)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DROP_COLS = ["label", "difficulty", "attack_category"]  # colonnes non-features


def preprocess(train_path, test_path):
    train, test = load_nsl_kdd(train_path, test_path)

    # ---------- 1. Encodage des variables catégorielles ----------
    # On fit les encoders sur le train, puis on les applique au test.
    # Important : le test peut contenir des catégories absentes du train
    # (ex: nouveaux services) -> on les gère avec un fallback.
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        train[col + "_enc"] = le.fit_transform(train[col])

        # Gérer les catégories inconnues dans le test
        known_classes = set(le.classes_)
        test[col] = test[col].apply(lambda x: x if x in known_classes else le.classes_[0])
        test[col + "_enc"] = le.transform(test[col])

        encoders[col] = le

    # ---------- 2. Préparation des labels ----------
    # Binaire : normal (0) vs attack (1)
    train["label_binary"] = (train["label"] != "normal").astype(int)
    test["label_binary"] = (test["label"] != "normal").astype(int)

    # Multi-classe : normal / dos / probe / r2l / u2r
    category_encoder = LabelEncoder()
    train["label_multiclass"] = category_encoder.fit_transform(train["attack_category"])
    test["attack_category"] = test["attack_category"].apply(
        lambda x: x if x in category_encoder.classes_ else "normal"
    )
    test["label_multiclass"] = category_encoder.transform(test["attack_category"])

    # ---------- 3. Construction de X (features) ----------
    feature_cols = [c for c in train.columns if c not in DROP_COLS + CATEGORICAL_COLS
                     and not c.startswith("label")]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    # ---------- 4. Normalisation ----------
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    y_train_bin, y_test_bin = train["label_binary"], test["label_binary"]
    y_train_multi, y_test_multi = train["label_multiclass"], test["label_multiclass"]

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train_bin": y_train_bin,
        "y_test_bin": y_test_bin,
        "y_train_multi": y_train_multi,
        "y_test_multi": y_test_multi,
        "feature_cols": feature_cols,
        "category_encoder": category_encoder,
        "scaler": scaler,
    }


if __name__ == "__main__":
    data = preprocess(
        os.path.join(DATA_DIR, "KDDTrain+.txt"),
        os.path.join(DATA_DIR, "KDDTest+.txt"),
    )

    print("Nombre de features utilisées :", len(data["feature_cols"]))
    print("\nShape X_train :", data["X_train"].shape)
    print("Shape X_test  :", data["X_test"].shape)

    print("\nAperçu X_train (5 premières lignes, 6 premières colonnes) :")
    print(data["X_train"].iloc[:5, :6])

    print("\nRépartition y_train (binaire) :")
    print(data["y_train_bin"].value_counts())

    print("\nRépartition y_train (multi-classe) :")
    classes = data["category_encoder"].classes_
    counts = data["y_train_multi"].value_counts().sort_index()
    for idx, count in counts.items():
        print(f"  {classes[idx]:8s} -> {count}")
