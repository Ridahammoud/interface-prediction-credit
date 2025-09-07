import streamlit as st
import requests

# URL de ton API Render
API_URL = "https://home-credit-predictor-g51k.onrender.com"

st.title("Home Credit Predictor - Batch CSV")

uploaded_file = st.file_uploader("Choisir un fichier CSV pour batch prediction", type="csv")

if uploaded_file is not None:
    st.write(f"Fichier sélectionné : {uploaded_file.name}")
    
    if st.button("Envoyer à l'API"):
        # Envoyer le fichier à l'endpoint HTML de Render
        files = {"file": uploaded_file.getvalue()}  # bytes
        try:
            response = requests.post(f"{API_URL}/predict_batch_upload", files={"file": uploaded_file})
            st.write("Status code:", response.status_code)
            
            # Comme l'API renvoie du HTML, on peut l'afficher en markdown ou raw
            st.subheader("Réponse de l'API (HTML brut)")
            st.code(response.text, language="html")
            
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")
