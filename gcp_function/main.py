import functions_framework
import pandas as pd
import os
import json

# 1. Chargement des CSV au démarrage du conteneur (exécuté une seule fois par instance)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH_COLLAB = os.path.join(BASE_DIR, "data_api", "df_matrice_recommandation_collaborative_filtering.csv")
CSV_PATH_CONTENT = os.path.join(BASE_DIR, "data_api", "matrix_recommendation_article_for_content_based.csv")

try:
    df_matrix_collaborative = pd.read_csv(CSV_PATH_COLLAB, index_col=0)
    print(f"Matrice collaborative chargée ({len(df_matrix_collaborative)} utilisateurs).")
except Exception as e:
    print(f"Erreur chargement collab: {e}")
    df_matrix_collaborative = None

try:
    df_matrix_content = pd.read_csv(CSV_PATH_CONTENT, index_col=0)
    print(f"Matrice content-based chargée ({len(df_matrix_content)} utilisateurs).")
except Exception as e:
    print(f"Erreur chargement content: {e}")
    df_matrix_content = None


@functions_framework.http
def recommend(request):
    """
    Point d'entrée de la Cloud Function.
    Prend en paramètre un objet flask.Request
    """
    # GESTION DU CORS (Important pour appeler l'API depuis un site web)
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json'}

    # VERIFICATION MATRICES
    if df_matrix_collaborative is None or df_matrix_content is None:
        return (json.dumps({"error": "Problème interne : impossible de charger les matrices."}), 500, headers)

    # RÉCUPÉRATION DES PARAMÈTRES D'URL (ex: ?user_id=123&is_collaborative=false)
    user_id_str = request.args.get('user_id')
    is_collaborative_str = request.args.get('is_collaborative', 'true').lower()
    
    # VALIDATION
    if not user_id_str:
        return (json.dumps({"error": "Le paramètre user_id est requis. (ex: ?user_id=0)"}), 400, headers)
        
    try:
        user_id = int(user_id_str)
    except ValueError:
        return (json.dumps({"error": "user_id doit être un nombre entier."}), 400, headers)

    is_collaborative = (is_collaborative_str == 'true')
    
    # LOGIQUE DE RECOMMANDATION
    matrix_to_use = df_matrix_collaborative if is_collaborative else df_matrix_content
    model_used = "collaborative_filtering" if is_collaborative else "content_based"
    
    if user_id not in matrix_to_use.index:
        return (json.dumps({"error": f"Utilisateur {user_id} introuvable dans la matrice {model_used}."}), 404, headers)
    
    # On prend uniquement les 5 derniers éléments pour gérer le format de chaque matrice
    recommended_items = matrix_to_use.loc[user_id].tolist()[-5:]
    
    response = {
        "user_id": user_id,
        "model": model_used,
        "recommendations": [int(item) for item in recommended_items]
    }
    
    # Retourner un Tuple : (Données String/JSON, Statut HTTP, Headers)
    return (json.dumps(response), 200, headers)
