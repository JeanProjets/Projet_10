# Content bases recommendation

On regarde juste le contenue des articles (embeddings) et on recommande les articles les plus proches en terme de similarité (distance cosinus). 

# Collaborative filtering recommendation

On regarde les préférences des utilisateurs et on recommande les articles les plus aimés par les utilisateurs similaires.

# Pour run l'API FASTApi
uvicorn main:app --reload

être sûr que j'ai : pip install fastapi uvicorn pandas

Pour tester l'appli sur Google Cloud Function

En local (FastAPI), c'était dans le chemin de l'URL : .../recommend/0
Sur Google Cloud Functions, l'URL est fixe, on utilise donc ce qu'on appelle des paramètres de requête (Query Parameters), avec un point d'interrogation ?. C'est exactement ce que dit le message d'erreur !
Essayez de cliquer sur ces liens corrigés :
1. Pour le filtrage collaboratif (par défaut) : 👉 https://us-central1-projet-10-498214.cloudfunctions.net/api-recommandation?user_id=0

2. Pour le Content-Based : 👉 https://us-central1-projet-10-498214.cloudfunctions.net/api-recommandation?user_id=0&is_collaborative=false

(Vous pouvez changer le 0 à la fin de l'URL par n'importe quel ID d'utilisateur pour tester, par exemple ?user_id=42)