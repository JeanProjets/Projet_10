"""
Interface Streamlit "dumb" — Frontend local pour le système de recommandation.
Appelle l'API Flask pour obtenir les recommandations.
"""
import streamlit as st
import requests
import os

# Configuration
FLASK_API_URL = os.environ.get("FLASK_API_URL", "http://localhost:5050")
CLOUD_FUNCTION_URL = os.environ.get("CLOUD_FUNCTION_URL", "")

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

# --- Mode info ---
if CLOUD_FUNCTION_URL:
    st.success(f"✅ **Mode Cloud** — Connecté à la Cloud Function")
    api_base = CLOUD_FUNCTION_URL
    use_cloud = True
else:
    st.info("🔧 **Mode Local** — Utilisation de l'API Flask locale")
    api_base = FLASK_API_URL
    use_cloud = False

# --- Récupérer la liste des utilisateurs ---
st.markdown("### 👤 Sélectionner un utilisateur")

# Liste de user_ids pour la démo
user_ids = list(range(0, 20))

# Essayer de récupérer depuis l'API Flask
if not use_cloud:
    try:
        resp = requests.get(f"{FLASK_API_URL}/users", timeout=5)
        if resp.status_code == 200:
            user_ids = resp.json().get("user_ids", user_ids)
    except requests.exceptions.RequestException:
        st.warning("⚠️ API Flask non disponible. Utilisation de la liste par défaut.")

selected_user = st.selectbox(
    "Choisir un user_id :",
    options=user_ids,
    index=0,
    help="Sélectionnez un identifiant utilisateur pour voir ses recommandations.",
)

# --- Bouton de recommandation ---
st.markdown("### 📰 Articles Recommandés")

if st.button("🔍 Obtenir les recommandations", type="primary", use_container_width=True):
    with st.spinner("Chargement des recommandations..."):
        try:
            if use_cloud:
                # Appel direct à la Cloud Function
                response = requests.get(
                    CLOUD_FUNCTION_URL,
                    params={"user_id": selected_user},
                    timeout=15,
                )
            else:
                # Appel à l'API Flask locale
                response = requests.get(
                    f"{FLASK_API_URL}/recommend/{selected_user}",
                    timeout=10,
                )

            if response.status_code == 200:
                data = response.json()
                recommendations = data.get("recommendations", [])
                model_version = data.get("model_version", "inconnu")
                message = data.get("message", "")

                # Afficher les résultats
                st.markdown(f"**Modèle** : `{model_version}`")
                if message:
                    st.caption(message)

                st.markdown("---")

                for i, article in enumerate(recommendations, 1):
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.markdown(f"### {i}")
                    with col2:
                        article_title = article.get('title', f"Article #{article['article_id']}")
                        st.markdown(f"**{article_title}**")
                        st.caption(
                            f"📄 ID: {article['article_id']} | "
                            f"🏷️ Catégorie: {article.get('category', 'N/A')}"
                        )
                    st.markdown("---")

                st.success(f"✅ {len(recommendations)} articles recommandés pour l'utilisateur {selected_user}")
            else:
                st.error(f"❌ Erreur {response.status_code} : {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Impossible de se connecter. "
                "Vérifiez que l'API Flask est lancée (`python app.py` dans flask_api/)."
            )
        except requests.exceptions.Timeout:
            st.error("⏳ Timeout — La requête a pris trop de temps.")
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {str(e)}")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "My Content MVP — Version de test (dumb) 🧪"
    "</div>",
    unsafe_allow_html=True,
)
