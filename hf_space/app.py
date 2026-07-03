import streamlit as st
import requests

# URL de votre Google Cloud Function déployée (vous pouvez la modifier ici si elle change)
CLOUD_FUNCTION_URL = "https://us-central1-projet-10-498214.cloudfunctions.net/api-recommandation"

# --- Page Config ---
st.set_page_config(
    page_title="My Content — Recommandations",
    page_icon="📚",
    layout="centered",
)

# --- Header ---
st.title("📚 My Content")
st.subheader("Système de Recommandation d'Articles")
st.markdown("---")

# --- Interface utilisateur ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Sélectionner un utilisateur")
    # L'utilisateur choisit un ID entre 0 et 1000 pour tester
    selected_user = st.number_input(
        "ID de l'utilisateur (ex: 0)", 
        min_value=0, 
        max_value=100000, 
        value=0, 
        step=1
    )

with col2:
    st.markdown("### ⚙️ Modèle de recommandation")
    # Choix entre Collaboratif et Content-based
    model_choice = st.radio(
        "Choisissez l'algorithme :",
        options=["Filtrage Collaboratif", "Content-Based"],
        index=0
    )

# Conversion du choix en booléen pour l'API
is_collaborative = True if model_choice == "Filtrage Collaboratif" else False

st.markdown("---")

# --- Bouton d'action ---
if st.button("🔍 Obtenir les recommandations", type="primary", use_container_width=True):
    with st.spinner("Génération des recommandations en cours... (Appel Serverless)"):
        try:
            # Paramètres de la requête
            params = {
                "user_id": selected_user,
                "is_collaborative": str(is_collaborative).lower() # 'true' ou 'false'
            }
            
            # Appel à la Cloud Function
            response = requests.get(CLOUD_FUNCTION_URL, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get("recommendations", [])
                model_used = data.get("model", "inconnu")

                # Affichage des résultats
                st.success(f"✅ Recommandations générées avec succès via **{model_used}**")
                
                st.markdown(f"### Top 5 des articles pour l'utilisateur {selected_user} :")
                
                for i, article_id in enumerate(recommendations, 1):
                    # Affichage visuel amélioré
                    st.info(f"**Recommandation #{i}** : Article ID `{article_id}`")
                    
            elif response.status_code == 404:
                st.warning(f"⚠️ L'utilisateur {selected_user} n'a pas été trouvé dans la matrice de données.")
            else:
                st.error(f"❌ Erreur de l'API (Code {response.status_code}) : {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Impossible de se connecter à la Google Cloud Function. Erreur : {e}")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "My Content MVP — Démo Serverless Google Cloud Functions"
    "</div>",
    unsafe_allow_html=True,
)
