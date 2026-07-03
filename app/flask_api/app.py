"""
API FastAPI — Backend local pour la recommandation
"""
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API de Recommandation")

# Configuration CORS pour autoriser les requêtes d'un frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Construit le chemin absolu vers les CSV pour éviter les problèmes de dossier courant
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH_COLLAB = os.path.join(BASE_DIR, "data_api", "df_matrice_recommandation_collaborative_filtering.csv")
CSV_PATH_CONTENT = os.path.join(BASE_DIR, "data_api", "matrix_recommendation_article_for_content_based.csv")

try:
    df_matrix_collaborative = pd.read_csv(CSV_PATH_COLLAB, index_col=0)
    print(f"Matrice collaborative chargée avec succès ({len(df_matrix_collaborative)} utilisateurs).")
except FileNotFoundError:
    print(f"Erreur : Impossible de trouver le fichier {CSV_PATH_COLLAB}")
    df_matrix_collaborative = None

try:
    df_matrix_content = pd.read_csv(CSV_PATH_CONTENT, index_col=0)
    print(f"Matrice content-based chargée avec succès ({len(df_matrix_content)} utilisateurs).")
except FileNotFoundError:
    print(f"Erreur : Impossible de trouver le fichier {CSV_PATH_CONTENT}")
    df_matrix_content = None


@app.get("/")
def index():
    """Health check."""
    return {
        "status": "ok",
        "service": "Recommendation API (FastAPI)",
        "version": "fastapi_v1",
    }


@app.get("/recommend/{user_id}")
async def get_recommendation(
    user_id: int, 
    is_collaborative: bool = Query(True, description="Si True, utilise le filtrage collaboratif. Sinon, utilise le content-based.")
):
    """
    Retourne les 5 articles recommandés pour un utilisateur en choisissant la matrice.
    """
    # Sélection de la matrice
    matrix_to_use = df_matrix_collaborative if is_collaborative else df_matrix_content
    model_used = "collaborative_filtering" if is_collaborative else "content_based"
    
    if matrix_to_use is None:
        raise HTTPException(status_code=500, detail=f"La matrice de recommandation pour {model_used} n'a pas pu être chargée.")
    
    if user_id not in matrix_to_use.index:
        raise HTTPException(status_code=404, detail=f"Utilisateur {user_id} introuvable dans la matrice {model_used}.")
    
    # On prend uniquement les 5 derniers éléments [-5:]
    # Cela permet de gérer le fait que la matrice content-based contient
    # l'ID utilisateur dans la première colonne de données, contrairement à la collaborative.
    recommended_items = matrix_to_use.loc[user_id].tolist()[-5:]
    
    return {
        "user_id": user_id,
        "model": model_used,
        "recommendations": [int(item) for item in recommended_items]
    }

# Si on lance le script directement avec python app.py
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 API FastAPI démarrée sur http://0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
