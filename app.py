import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ===================== PAGE D'ACCUEIL =====================
st.set_page_config(page_title="SVSA - Système de Veille Sanitaire", layout="wide")

st.title("SVSA : Système de Veille Sanitaire et d'Aide à la Décision en Temps Réel")

st.markdown("""
### Le Problème
La santé publique souffre d’un manque criant de **données unifiées et en temps réel**.  
Cela entraîne :
- des **retards critiques dans le diagnostic**,  
- une **difficulté à anticiper les épidémies régionales**,  
- et une **gestion réactive** plutôt que préventive des crises sanitaires.

---

### La Solution : Une Plateforme Intelligente à Deux Niveaux
1 **Application mobile pour les agents de santé communautaire**  
→ Une interface simple d’**aide à la décision médicale** permettant de :
- Saisir les consultations (exemple : paludisme)
- Standardiser le diagnostic
- Recommander un traitement rapide selon les protocoles  

Exemple : module de **registre médical** (voir ci-dessous).

2 **Tableau de bord centralisé pour les autorités sanitaires**  
→ Vue consolidée et en temps réel des diagnostics collectés depuis les postes et centres de santé :
- Analyse automatique des tendances  
- Détection précoce des foyers épidémiques  
- Alerte des autorités sanitaires pour **une action rapide**.

---

### L'Impact
Ce système transforme la **gestion de crise** en **prévention proactive** :
- Diagnostic et traitement plus rapides  
- Qualité des soins primaires renforcée  
- Surveillance régionale en temps réel  
- Gouvernance sanitaire basée sur la donnée (**Data-Driven Health**)

---
""")

st.markdown("""
### Accès au registre médical
Cliquez sur le bouton ci-dessous pour ouvrir l’application du registre médical (exemple : **Diagnostic Paludisme**).
""")

st.markdown(
    """
    <a href="https://projet-covath-on.vercel.app/" target="_blank">
        <button style="
            background-color:#007BFF;
            color:white;
            border:none;
            padding:10px 20px;
            border-radius:8px;
            font-size:16px;
            cursor:pointer;">
            Ouvrir le Registre Médical
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

st.markdown("""
---
### Tableau de bord épidémiologique
Analyse en temps réel des données de consultation et prédiction des cas futurs grâce à **ARIMA** ou **Random Forest**.
""")


# ===================== CHARGEMENT DES DONNÉES =====================
@st.cache_data
def load_data(path="data/Base_General.xlsx"):
    df = pd.read_excel(path)

    # Normalisation des noms de colonnes
    df.columns = df.columns.str.strip().str.lower()

    # Vérifie et convertit la colonne de date
    date_col = [col for col in df.columns if "date" in col]
    if date_col:
        df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors="coerce")
        df.rename(columns={date_col[0]: "date"}, inplace=True)
    else:
        st.warning(" Aucune colonne de type 'date' trouvée.")
        df["date"] = pd.NaT

    # Créer des tranches d'âge
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=[0,5,15,30,60,100],
            labels=["0-5","6-15","16-30","31-60","60+"],
            include_lowest=True
        )

    return df

df = load_data()

# ===================== BARRE LATÉRALE =====================
st.sidebar.header("Filtres interactifs")

colonnes_possibles = df.columns.tolist()

# Les colonnes suivantes peuvent ne pas exister, donc on les gère dynamiquement
if "localité" in colonnes_possibles:
    localites = ["Tous"] + sorted(df["localité"].dropna().unique().tolist())
else:
    localites = ["Tous"]

if "sexe" in colonnes_possibles:
    sexes = sorted(df["sexe"].dropna().unique().tolist())
else:
    sexes = []

selected_localite = st.sidebar.selectbox("Localité :", localites)
selected_sexe = st.sidebar.multiselect("Sexe :", sexes, default=sexes)
date_range = st.sidebar.date_input("Période :", [])

# ===================== FILTRAGE =====================
df_filtered = df.copy()

if selected_localite != "Tous":
    df_filtered = df_filtered[df_filtered["localité"] == selected_localite]
if selected_sexe:
    df_filtered = df_filtered[df_filtered["sexe"].isin(selected_sexe)]
if len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered["date"] >= pd.to_datetime(date_range[0])) &
        (df_filtered["date"] <= pd.to_datetime(date_range[1]))
    ]

# ===================== INDICATEURS CLÉS =====================
st.markdown("## Statistiques générales")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Nombre total de consultations", len(df_filtered))
col2.metric("Taux de paludisme", f"{(df_filtered['palu'].str.contains('oui', case=False).mean()*100):.1f}%" if "palu" in df.columns else "N/A")
col3.metric("Âge moyen", f"{df_filtered['age'].mean():.1f} ans" if "age" in df.columns else "N/A")
col4.metric("Température moyenne", f"{df_filtered['temperature (t°c)'].mean():.1f} °C" if "temperature (t°c)" in df.columns else "N/A")

# ===================== VISUALISATIONS =====================
st.markdown("## Analyses descriptives")

colA, colB = st.columns(2)

if "sexe" in df.columns and "palu" in df.columns:
    with colA:
        fig1 = px.histogram(df_filtered, x="sexe", color="palu", barmode="group",
                            title="Répartition par sexe et présence du paludisme")
        st.plotly_chart(fig1, use_container_width=True)


if "age_group" in df.columns:
    with colB:
        # Comptage et renommage propre
        df_age = df_filtered["age_group"].value_counts().reset_index()
        df_age.columns = ["Tranche d'âge", "Nombre de cas"]

        # Création du graphique
        fig2 = px.bar(
            df_age,
            x="Tranche d'âge",
            y="Nombre de cas",
            color="Tranche d'âge",
            title="Distribution par tranche d’âge"
        )

        st.plotly_chart(fig2, use_container_width=True)

# ===================== ÉVOLUTION TEMPORELLE =====================
st.markdown("## Évolution des cas dans le temps")

if "palu" in df.columns:
    df_time = df_filtered.groupby("date")["palu"].apply(lambda x: (x.str.contains("oui", case=False)).sum()).reset_index(name="Cas")
    fig3 = px.line(df_time, x="date", y="Cas", markers=True, title="Évolution des cas de paludisme")
    st.plotly_chart(fig3, use_container_width=True)

# ===================== PRÉDICTION ARIMA =====================
st.markdown("## Prédiction ARIMA des cas futurs")

if "palu" in df.columns:
    ts = df_time.set_index("date")["Cas"].asfreq('W', fill_value=0)

    if len(ts) > 10:
        if st.button("Lancer la prédiction ARIMA"):
            model = ARIMA(ts, order=(1,1,1))
            model_fit = model.fit()
            fc = model_fit.get_forecast(steps=8)
            pred = fc.predicted_mean
            fig, ax = plt.subplots()
            ax.plot(ts.index, ts.values, label="Historique")
            ax.plot(pred.index, pred.values, label="Prévision", linestyle="--")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("Données insuffisantes pour lancer la prédiction.")

# ===================== PREDICTION INDIVIDUELLE DU PALUDISME =====================
st.markdown("##  Détection du paludisme à partir des paramètres médicaux")

# Colonnes d’intérêt
features = ["Age", "Temperature (T°c)", "Poids", "P_Systolique", "P_diastolique", "Taux", "GC"]
target = "Palu"

# Vérifie la présence des colonnes
if all(col.lower() in [c.lower() for c in df.columns] for col in features + [target]):

    # Harmonisation des noms
    df.columns = df.columns.str.strip().str.lower()
    features_lower = [f.lower() for f in features]
    target_lower = target.lower()

    # Préparation du dataset
    df_model = df.dropna(subset=features_lower + [target_lower]).copy()
    df_model[target_lower] = (
        df_model[target_lower]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({'oui': 1, 'non': 0, '1': 1, '0': 0})
        .fillna(0)
    )

    # Vérifie la distribution des classes
    st.write("###  Distribution des classes (Palu)")
    st.write(df_model[target_lower].value_counts())

    # Vérifie qu’il y a au moins 2 classes
    unique_classes = df_model[target_lower].unique()

    if len(unique_classes) > 1:
        # Équilibrage uniquement si les deux classes existent
        from sklearn.utils import resample
        count_class_0 = len(df_model[df_model[target_lower] == 0])
        count_class_1 = len(df_model[df_model[target_lower] == 1])
        max_count = max(count_class_0, count_class_1)

        df_balanced = pd.concat([
            resample(df_model[df_model[target_lower] == 1],
                     replace=True,
                     n_samples=max_count if count_class_1 > 0 else count_class_0,
                     random_state=42),
            resample(df_model[df_model[target_lower] == 0],
                     replace=True,
                     n_samples=max_count if count_class_0 > 0 else count_class_1,
                     random_state=42)
        ])
    else:
        # Cas où il n’y a qu’une seule classe
        st.warning(" Votre base ne contient qu’une seule classe de 'Palu' (tous oui ou tous non). Pas d’équilibrage possible.")
        df_balanced = df_model.copy()

    # Prépare X et y
    X = df_balanced[features_lower]
    y = df_balanced[target_lower]

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f" Précision du modèle : {acc*100:.2f}%")

    # ------------------ Simulation interactive ------------------
    st.markdown("###  Simulation d’un nouveau diagnostic")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Âge", 0, 100, 25)
        temp = st.number_input("Température (°C)", 30.0, 43.0, 37.0)
        poids = st.number_input("Poids (kg)", 10, 120, 60)
    with col2:
        p_systolique = st.number_input("Pression systolique", 80, 200, 120)
        p_diastolique = st.number_input("Pression diastolique", 50, 120, 80)
        taux = st.number_input("Taux (HB ou autre)", 0.0, 20.0, 10.0)
    with col3:
        gc = st.number_input("Glycémie capillaire (g/L)", 0.0, 3.0, 1.0)

    if st.button("Diagnostiquer maintenant"):
        data_new = pd.DataFrame(
            [[age, temp, poids, p_systolique, p_diastolique, taux, gc]],
            columns=features_lower
        )

        pred = model.predict(data_new)[0]
        # Gestion robuste des probabilités
        if hasattr(model, "predict_proba"):
            if len(model.classes_) == 2:
                proba = model.predict_proba(data_new)[0][1]
            else:
                proba = model.predict_proba(data_new)[0][0] if model.classes_[0] == 1 else 0.0
        else:
            proba = 0.5  # Valeur neutre par défaut

        # Résultat
        if pred == 1:
            st.error(f" Risque élevé de paludisme détecté ({proba*100:.1f}% de probabilité)")
        else:
            st.success(f" Aucun signe de paludisme détecté ({(1-proba)*100:.1f}% de probabilité)")

else:
    st.warning(" Certaines colonnes nécessaires au modèle sont absentes du fichier Excel.")

# ===================== PIED DE PAGE =====================
st.markdown("---")
st.markdown("© 2025 **SVSA - HealthWatch** | Prototype de Veille Sanitaire et Prédiction du Paludisme.")
