import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
st.set_page_config(page_title="HealthWatch MVP", layout="wide")
st.title(" HealthWatch - MVP de Surveillance Épidémiologique")

# ===================== CHARGEMENT DES DONNÉES =====================
@st.cache_data
def load_data(path="data/Base_General.xlsx"):
    df = pd.read_excel(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["age_group"] = pd.cut(df["age"], bins=[0,5,15,30,60,100],
                             labels=["0-5","6-15","16-30","31-60","60+"],
                             include_lowest=True)
    return df

df = load_data()

# ===================== BARRE LATÉRALE =====================
st.sidebar.header(" Filtres interactifs")

districts = ["Tous"] + sorted(df["district"].dropna().unique().tolist())
centres = ["Tous"] + sorted(df["centre"].dropna().unique().tolist())
types_palu = sorted(df["type_palu"].dropna().unique().tolist()) if "type_palu" in df.columns else []
sexes = sorted(df["sexe"].dropna().unique().tolist()) if "sexe" in df.columns else []

selected_district = st.sidebar.selectbox("District :", districts)
selected_centre = st.sidebar.selectbox("Centre de santé :", centres)
selected_sexe = st.sidebar.multiselect("Sexe :", sexes, default=sexes)
selected_type = st.sidebar.multiselect("Type de Palu :", types_palu, default=types_palu)
date_range = st.sidebar.date_input("Plage de dates :", [])

# ===================== APPLICATION DES FILTRES =====================
df_filtered = df.copy()

if selected_district != "Tous":
    df_filtered = df_filtered[df_filtered["district"] == selected_district]

if selected_centre != "Tous":
    df_filtered = df_filtered[df_filtered["centre"] == selected_centre]

if selected_sexe:
    df_filtered = df_filtered[df_filtered["sexe"].isin(selected_sexe)]

if selected_type:
    df_filtered = df_filtered[df_filtered["type_palu"].isin(selected_type)]

if len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered["date"] >= pd.to_datetime(date_range[0])) &
        (df_filtered["date"] <= pd.to_datetime(date_range[1]))
    ]

# ===================== INDICATEURS CLÉS =====================
st.markdown("##  Statistiques générales")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Nombre total de cas", len(df_filtered))
col2.metric("Cas graves", len(df_filtered[df_filtered["type_palu"]=="Grave"]) if "type_palu" in df.columns else 0)
col3.metric("Âge moyen", f"{df_filtered['age'].mean():.1f} ans" if "age" in df.columns else "N/A")
col4.metric("Température moyenne", f"{df_filtered['temperature'].mean():.1f} °C" if "temperature" in df.columns else "N/A")

# ===================== VISUALISATIONS =====================
st.markdown("##  Analyses descriptives interactives")

colA, colB = st.columns(2)

# Répartition par sexe et type de palu
if "sexe" in df.columns and "type_palu" in df.columns:
    with colA:
        fig1 = px.histogram(df_filtered, x="sexe", color="type_palu", barmode="group",
                            title="Répartition par sexe et type de palu")
        st.plotly_chart(fig1, use_container_width=True)

# Répartition par tranche d’âge
if "age_group" in df.columns:
    with colB:
        fig2 = px.bar(df_filtered["age_group"].value_counts().reset_index(),
                      x="index", y="age_group",
                      title="Distribution par tranche d’âge",
                      labels={"index":"Tranche d'âge", "age_group":"Nombre de cas"})
        st.plotly_chart(fig2, use_container_width=True)

# Évolution temporelle des cas
st.markdown("###  Évolution des cas dans le temps")
df_time = df_filtered.groupby("date").size().reset_index(name="Cas")
fig3 = px.line(df_time, x="date", y="Cas", markers=True, title="Évolution hebdomadaire/mensuelle des cas")
st.plotly_chart(fig3, use_container_width=True)

# ===================== ANALYSE PRÉDICTIVE (ARIMA) =====================
st.markdown("##  Prédiction ARIMA des cas futurs")

ts = df_filtered.groupby("date")["cas"].sum().asfreq('W', fill_value=0)

if len(ts) > 10:
    if st.button("Lancer la prédiction ARIMA"):
        model = ARIMA(ts, order=(1,1,1))
        model_fit = model.fit()
        fc = model_fit.get_forecast(steps=8)
        pred = fc.predicted_mean
        # Affichage
        fig, ax = plt.subplots()
        ax.plot(ts.index, ts.values, label="Historique")
        ax.plot(pred.index, pred.values, label="Prévision", linestyle="--")
        ax.legend()
        st.pyplot(fig)
else:
    st.info("Données insuffisantes pour lancer la prédiction (minimum 10 points).")
    
# ===================== PREDICTION INDIVIDUELLE DU PALUDISME =====================
st.markdown("## Détection du paludisme à partir des paramètres médicaux")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Colonnes pertinentes
features = ["Age", "Temperature (T°c)", "Poids", "P_Systolique", "P_diastolique", "Taux", "GC"]
target = "Palu"

# Vérifie que toutes les colonnes existent
if all(col in df.columns for col in features + [target]):

    # Nettoyage de la base
    df_model = df.dropna(subset=features + [target])
    df_model = df_model.copy()
    df_model[target] = df_model[target].astype(str).str.strip().str.lower().map({
        'oui': 1, 'non': 0, '1': 1, '0': 0
    }).fillna(0)  # encode Palu en 0/1

    X = df_model[features]
    y = df_model[target]

    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Précision du modèle :** {acc*100:.2f}%")

    st.markdown("### Simulation d’un nouveau diagnostic")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Âge", 0, 100, 25)
        temp = st.number_input("Température (°C)", 30.0, 43.0, 37.0)
        poids = st.number_input("Poids (kg)", 10, 120, 60)
    with col2:
        ps = st.number_input("Pression systolique", 80, 200, 120)
        pd = st.number_input("Pression diastolique", 50, 120, 80)
        taux = st.number_input("Taux (HB ou autre)", 0.0, 20.0, 10.0)
    with col3:
        gc = st.number_input("Glycémie capillaire (g/L)", 0.0, 3.0, 1.0)

    if st.button("🔬 Diagnostiquer"):
        data_new = pd.DataFrame([[age, temp, poids, ps, pd, taux, gc]], columns=features)
        pred = model.predict(data_new)[0]
        proba = model.predict_proba(data_new)[0][1]

        if pred == 1:
            st.error(f" Risque élevé de paludisme détecté ({proba*100:.1f}% de probabilité)")
        else:
            st.success(f" Aucun signe de paludisme détecté ({proba*100:.1f}% de probabilité)")

else:
    st.warning("Certaines colonnes nécessaires au modèle de prédiction sont absentes de la base.")


# ===================== PIED DE PAGE =====================
st.markdown("---")
st.markdown(" *HealthWatch – Tableau de bord interactif et prédictif pour la surveillance du paludisme.*")
