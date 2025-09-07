# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import requests
import io

# URL de ton API Render
API_URL = "https://home-credit-predictor-g51k.onrender.com"

st.set_page_config(page_title="Home Credit Predictor", layout="wide")
st.title("Home Credit Predictor - Streamlit Interface")

# -------------------
# Prédiction individuelle
# -------------------
st.header("Prédiction individuelle")

EXT_SOURCE_1 = st.number_input("EXT_SOURCE_1", value=0.5)
AMT_CREDIT = st.number_input("AMT_CREDIT", value=100000)

threshold = st.number_input("Seuil décision (optionnel)", min_value=0.0, max_value=1.0, value=None)

if st.button("Prédire Individuel"):
    payload = {
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "AMT_CREDIT": AMT_CREDIT
    }
    if threshold is not None:
        payload["threshold"] = threshold

    try:
        resp = requests.post(f"{API_URL}/predict_single", json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            st.success("Prédiction reçue !")
            st.json(result)
        else:
            st.error(f"Erreur API: {resp.status_code} - {resp.text}")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API Render : {e}")

# -------------------
# Prédiction batch (CSV)
# -------------------
st.header("Prédiction batch (CSV)")

uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

batch_threshold = st.number_input("Seuil pour filtrer les résultats batch (optionnel)", min_value=0.0, max_value=1.0, value=None)

if uploaded_file is not None and st.button("Envoyer CSV pour prédiction"):
    try:
        files = {"file": uploaded_file}
        params = {}
        if batch_threshold is not None:
            params["seuil"] = batch_threshold

        resp = requests.post(f"{API_URL}/predict_batch", files=files, params=params, timeout=60)

        if resp.status_code == 200:
            result_json = resp.json()
            st.success(f"Prédictions reçues ! {result_json['n_predictions']} sur {result_json['n_input_rows']} lignes")
            
            df_result = pd.DataFrame(result_json["predictions"])
            st.dataframe(df_result)

            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger les résultats CSV", csv_bytes, file_name="predictions_home_credit.csv")

        else:
            st.error(f"Erreur API: {resp.status_code} - {resp.text}")

    except Exception as e:
        st.error(f"Erreur lors de l'appel API Render : {e}")
