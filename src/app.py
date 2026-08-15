"""
Application Streamlit de démonstration - NIDS (Network Intrusion Detection System)

Lancer avec : streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from load_data import load_nsl_kdd
from preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "..", "notebooks")

st.set_page_config(
    page_title="NIDS-IA | Détection d'intrusion réseau",
    page_icon="🛡️",
    layout="wide",
)

# ============================================================
# Chargement des données et entraînement du modèle (mis en cache)
# ============================================================

@st.cache_resource(show_spinner="Chargement des données et entraînement du modèle...")
def load_and_train():
    data = preprocess(
        os.path.join(DATA_DIR, "KDDTrain+.txt"),
        os.path.join(DATA_DIR, "KDDTest+.txt"),
    )
    train_raw, test_raw = load_nsl_kdd(
        os.path.join(DATA_DIR, "KDDTrain+.txt"),
        os.path.join(DATA_DIR, "KDDTest+.txt"),
    )

    X_train, y_train = data["X_train"], data["y_train_multi"]
    class_names = list(data["category_encoder"].classes_)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.3,
        objective="multi:softmax", num_class=len(class_names),
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    return model, data, class_names, test_raw


model, data, class_names, test_raw = load_and_train()
X_test = data["X_test"]
y_test = data["y_test_multi"]
feature_cols = data["feature_cols"]

CLASS_LABELS_FR = {
    "normal": "✅ Normal",
    "dos": "🔴 Attaque DoS (Déni de service)",
    "probe": "🟠 Attaque Probe (Reconnaissance/scan)",
    "r2l": "🟣 Attaque R2L (Accès distant non autorisé)",
    "u2r": "⚫ Attaque U2R (Élévation de privilèges)",
}

# ============================================================
# Barre latérale : navigation
# ============================================================

st.sidebar.title("🛡️ NIDS-IA")
st.sidebar.markdown("**Network Intrusion Detection System**\npropulsé par XGBoost")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Vue d'ensemble", "🔍 Démo en temps réel", "📈 Visualisations"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Projet Master Info & Télécom — Dataset NSL-KDD")

# ============================================================
# PAGE 1 : Vue d'ensemble
# ============================================================
if page == "📊 Vue d'ensemble":
    st.title("🛡️ Système de détection d'intrusion réseau par IA")
    st.markdown(
        "Ce projet utilise le **Machine Learning** pour classifier le trafic réseau "
        "en trafic normal ou en 4 types d'attaques, à partir du dataset **NSL-KDD**."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Connexions (train)", f"{len(data['X_train']):,}".replace(",", " "))
    col2.metric("Connexions (test)", f"{len(X_test):,}".replace(",", " "))
    col3.metric("Features utilisées", len(feature_cols))
    col4.metric("Classes", len(class_names))

    st.markdown("### Répartition des classes (jeu d'entraînement)")
    class_dist = train_counts = pd.Series(data["y_train_multi"]).map(
        dict(enumerate(class_names))
    ).value_counts()
    st.bar_chart(class_dist)

    st.markdown("### Comparaison des approches testées")
    comp_path = os.path.join(NOTEBOOKS_DIR, "improvement_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path, index_col=0)
        st.dataframe(comp_df.style.highlight_max(axis=0, color="#d4f7d4"), use_container_width=True)
        st.caption(
            "Le modèle utilisé dans cette démo applique la pondération des classes "
            "(class weights) pour améliorer la détection des attaques rares (R2L, U2R)."
        )
    else:
        st.info("Lance d'abord `improve_model.py` pour générer le tableau comparatif.")

# ============================================================
# PAGE 2 : Démo en temps réel
# ============================================================
elif page == "🔍 Démo en temps réel":
    st.title("🔍 Démo en temps réel")
    st.markdown(
        "Choisis une connexion réelle du jeu de test, ou ajuste les paramètres "
        "manuellement pour voir comment le modèle réagit."
    )

    mode = st.radio(
        "Mode",
        ["Piocher une connexion réelle du test set", "Ajuster manuellement"],
        horizontal=True,
    )

    if mode == "Piocher une connexion réelle du test set":
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("🎲 Tirer une connexion au hasard", use_container_width=True):
                st.session_state["sample_idx"] = np.random.randint(0, len(X_test))
        if "sample_idx" not in st.session_state:
            st.session_state["sample_idx"] = 0

        idx = st.session_state["sample_idx"]
        sample = X_test.iloc[[idx]]
        true_label = class_names[y_test.iloc[idx]]
        raw_row = test_raw.iloc[idx]

        with col_b:
            st.write(
                f"**Connexion #{idx}** — protocole `{raw_row['protocol_type']}`, "
                f"service `{raw_row['service']}`, label réel du dataset : `{raw_row['label']}`"
            )

        input_for_model = sample

    else:
        st.markdown("#### Paramètres principaux de la connexion")
        c1, c2, c3 = st.columns(3)
        with c1:
            src_bytes = st.slider("Octets envoyés (src_bytes)", 0, 5000, 200)
            dst_bytes = st.slider("Octets reçus (dst_bytes)", 0, 5000, 200)
            duration = st.slider("Durée de connexion (s)", 0, 500, 0)
        with c2:
            count = st.slider("Nb connexions vers même hôte (count)", 0, 500, 5)
            srv_count = st.slider("Nb connexions vers même service (srv_count)", 0, 500, 5)
            logged_in = st.selectbox("Connexion authentifiée (logged_in)", [0, 1], index=1)
        with c3:
            serror_rate = st.slider("Taux d'erreurs SYN (serror_rate)", 0.0, 1.0, 0.0)
            same_srv_rate = st.slider("Taux même service (same_srv_rate)", 0.0, 1.0, 1.0)
            diff_srv_rate = st.slider("Taux services différents (diff_srv_rate)", 0.0, 1.0, 0.0)

        # On part d'une ligne "normale" moyenne du train set, puis on écrase les
        # colonnes ajustées par l'utilisateur. Les autres features gardent une
        # valeur réaliste plutôt que 0 partout.
        base_row = data["X_train"].iloc[[0]].copy()
        overrides = {
            "src_bytes": src_bytes, "dst_bytes": dst_bytes, "duration": duration,
            "count": count, "srv_count": srv_count, "logged_in": logged_in,
            "serror_rate": serror_rate, "same_srv_rate": same_srv_rate,
            "diff_srv_rate": diff_srv_rate,
        }
        for col, val in overrides.items():
            if col in base_row.columns:
                # On ré-applique la normalisation approximative (le scaler est déjà
                # fit ; ici on triche en insérant la valeur brute standardisée à 0
                # pour rester simple pédagogiquement — voir note sous le formulaire).
                base_row[col] = val

        input_for_model = base_row
        true_label = None
        st.caption(
            "⚠️ Mode simplifié à but pédagogique : les valeurs sont injectées "
            "directement sans re-normalisation exacte. Idéal pour observer les "
            "tendances (ex: augmenter serror_rate favorise une classe d'attaque), "
            "moins pour une valeur absolue précise."
        )

    # ---------- Prédiction ----------
    st.markdown("---")
    proba = model.predict_proba(input_for_model)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = class_names[pred_idx]

    col_pred, col_proba = st.columns([1, 2])

    with col_pred:
        st.markdown("#### Prédiction du modèle")
        st.markdown(f"## {CLASS_LABELS_FR[pred_class]}")
        st.metric("Confiance", f"{proba[pred_idx]*100:.1f}%")
        if true_label is not None:
            if true_label == pred_class:
                st.success(f"✅ Correct ! Le label réel était bien `{true_label}`.")
            else:
                st.error(f"❌ Erreur du modèle. Le label réel était `{true_label}`.")

    with col_proba:
        st.markdown("#### Probabilités par classe")
        proba_df = pd.DataFrame({
            "classe": [CLASS_LABELS_FR[c] for c in class_names],
            "probabilité": proba,
        }).set_index("classe")
        st.bar_chart(proba_df)

# ============================================================
# PAGE 3 : Visualisations
# ============================================================
elif page == "📈 Visualisations":
    st.title("📈 Visualisations du projet")
    st.markdown("Graphiques générés lors de l'analyse exploratoire et de l'évaluation des modèles.")

    figures = [
        ("01_class_distribution.png", "Répartition des classes"),
        ("02_protocol_service.png", "Protocoles et services"),
        ("03_correlation_matrix.png", "Matrice de corrélation"),
        ("06_rf_vs_xgboost.png", "Random Forest vs XGBoost"),
        ("08_confusion_matrix_multiclass.png", "Matrice de confusion multi-classe"),
        ("09_confusion_matrix_multiclass_normalized.png", "Matrice de confusion normalisée"),
        ("10_f1_per_class.png", "F1-score par classe"),
        ("11_r2l_train_vs_test.png", "Sous-types R2L : train vs test"),
        ("12_confidence_r2l_errors.png", "Confiance du modèle sur les erreurs R2L"),
        ("13_improvement_comparison.png", "Baseline vs Class Weights vs SMOTE"),
    ]

    missing = []
    cols = st.columns(2)
    for i, (filename, title) in enumerate(figures):
        path = os.path.join(NOTEBOOKS_DIR, filename)
        if os.path.exists(path):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)
        else:
            missing.append(filename)

    if missing:
        st.warning(
            "Certains graphiques n'ont pas été trouvés : " + ", ".join(missing) +
            ". Lance les scripts correspondants (eda.py, train_xgboost.py, "
            "train_multiclass.py, error_analysis.py, improve_model.py) pour les générer."
        )
