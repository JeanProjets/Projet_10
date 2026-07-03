"""
Cloud Function "dumb" pour tester le déploiement.
Cette version ne fait PAS de vraies recommandations.
Elle retourne simplement des articles aléatoires pour valider le pipeline de déploiement.
"""
import functions_framework
import json
import random


# Simule une "base de données" d'articles
DUMMY_ARTICLES = [
    {"article_id": 101, "title": "Introduction à Python", "category": "Tech"},
    {"article_id": 202, "title": "Les bases du Machine Learning", "category": "Data Science"},
    {"article_id": 303, "title": "Guide de voyage : Japon", "category": "Voyage"},
    {"article_id": 404, "title": "Recette de ramen maison", "category": "Cuisine"},
    {"article_id": 505, "title": "Les bienfaits du yoga", "category": "Bien-être"},
    {"article_id": 606, "title": "Comprendre la blockchain", "category": "Tech"},
    {"article_id": 707, "title": "Top 10 des films de 2025", "category": "Culture"},
    {"article_id": 808, "title": "L'art du minimalisme", "category": "Lifestyle"},
    {"article_id": 909, "title": "Débuter en photographie", "category": "Loisirs"},
    {"article_id": 1010, "title": "Histoire de l'intelligence artificielle", "category": "Tech"},
]


@functions_framework.http
def recommend_articles(request):
    """
    Cloud Function HTTP qui retourne 5 articles "recommandés" (aléatoires).
    
    Entrée (JSON ou query param) :
        - user_id : identifiant de l'utilisateur (int)
    
    Sortie (JSON) :
        - user_id : l'id reçu
        - recommendations : liste de 5 articles
        - model_version : "dumb_v1" (pour tracer quelle version est déployée)
    """
    # Gestion CORS (pour les appels depuis un navigateur)
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    headers = {"Access-Control-Allow-Origin": "*"}

    # Récupérer le user_id
    user_id = None

    # Essayer depuis le JSON body (POST)
    if request.is_json:
        data = request.get_json(silent=True)
        if data and "user_id" in data:
            user_id = data["user_id"]

    # Sinon depuis les query params (GET)
    if user_id is None:
        user_id = request.args.get("user_id")

    # Validation
    if user_id is None:
        return (
            json.dumps({"error": "Paramètre 'user_id' manquant. Utilise ?user_id=42 ou envoie un JSON."}),
            400,
            headers,
        )

    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return (
            json.dumps({"error": "Le user_id doit être un entier."}),
            400,
            headers,
        )

    # "Recommandation" : on tire 5 articles au hasard (déterministe par user_id pour la reproductibilité)
    random.seed(user_id)
    recommendations = random.sample(DUMMY_ARTICLES, 5)

    response = {
        "user_id": user_id,
        "recommendations": recommendations,
        "model_version": "dumb_v1",
        "message": "Ceci est une version de TEST. Les recommandations sont aléatoires.",
    }

    return (json.dumps(response, ensure_ascii=False), 200, headers)
