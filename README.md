# Content bases recommendation

On regarde juste le contenue des articles (embeddings) et on recommande les articles les plus proches en terme de similarité (distance cosinus). 

# Collaborative filtering recommendation

On regarde les préférences des utilisateurs et on recommande les articles les plus aimés par les utilisateurs similaires.

# Pour run l'API FASTApi
uvicorn main:app --reload

être sûr que j'ai : pip install fastapi uvicorn pandas
