import json

notebook_path = "notebooks/notebook_test.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

md_content = """## Étape suivante : Recommandation pour les utilisateurs (Version Optimisée avec Pandas)

Pour traiter l'ensemble des données rapidement (sur la variable `clicks`), on évite les boucles `for` en Python qui sont lentes.
On utilise plutôt les fonctionnalités natives de Pandas (`groupby`, `merge`, `melt`) qui sont écrites en C sous le capot et optimisées pour de gros volumes de données."""

md_cell = {
    "cell_type": "markdown",
    "id": "md_cell_reco_opti",
    "metadata": {},
    "source": [line + "\n" for line in md_content.split("\n")]
}
md_cell["source"][-1] = md_cell["source"][-1].rstrip("\n")

code_content = """import pandas as pd
import numpy as np
import random
import time

def generate_all_recommendations_optimized(clicks_df, matrix_reco_df, top_n=5):
    start_time = time.time()
    
    print("1. Tri des clics par timestamp...")
    # On trie pour avoir les plus récents en premier
    sorted_clicks = clicks_df.sort_values(by=['user_id', 'click_timestamp'], ascending=[True, False])
    
    print("2. Récupération des 5 derniers articles par utilisateur...")
    top_5_user_clicks = sorted_clicks.groupby('user_id').head(5)
    
    # On sauvegarde les articles lus pour les exclure plus tard
    user_read_articles = top_5_user_clicks.groupby('user_id')['click_article_id'].apply(set).to_dict()
    
    print("3. Jointure avec la matrice de recommandations...")
    merged = top_5_user_clicks.merge(matrix_reco_df, left_on='click_article_id', right_index=True, how='left')
    
    print("4. Préparation des recommandations...")
    # On transforme les colonnes reco_1 à reco_5 en lignes pour pouvoir compter
    reco_cols = ['reco_1', 'reco_2', 'reco_3', 'reco_4', 'reco_5']
    melted_recos = merged.melt(id_vars=['user_id'], value_vars=reco_cols, value_name='recommended_article_id')
    melted_recos = melted_recos.dropna(subset=['recommended_article_id'])
    # Conversion en entier (melt peut parfois transformer en float s'il y a des NaN)
    melted_recos['recommended_article_id'] = melted_recos['recommended_article_id'].astype(int)
    
    print("5. Comptage des occurrences (overlaps)...")
    reco_counts = melted_recos.groupby(['user_id', 'recommended_article_id']).size().reset_index(name='count')
    
    print("6. Filtrage des articles déjà lus...")
    read_articles_df = top_5_user_clicks[['user_id', 'click_article_id']].copy()
    read_articles_df['already_read'] = True
    
    filtered_recos = reco_counts.merge(
        read_articles_df, 
        left_on=['user_id', 'recommended_article_id'], 
        right_on=['user_id', 'click_article_id'], 
        how='left'
    )
    filtered_recos = filtered_recos[filtered_recos['already_read'].isnull()]
    
    print("7. Tri et sélection finale...")
    final_recos = filtered_recos.sort_values(by=['user_id', 'count'], ascending=[True, False])
    top_recos = final_recos.groupby('user_id').head(top_n)
    
    print("8. Création du dictionnaire de résultats...")
    user_recommendations_dict = top_recos.groupby('user_id')['recommended_article_id'].apply(list).to_dict()
    
    print("9. Remplissage au hasard pour les recommandations manquantes...")
    all_articles = list(matrix_reco_df.index)
    all_users = set(clicks_df['user_id'].unique())
    
    # Pour s'assurer que chaque utilisateur a bien 5 recommandations
    for user_id in all_users:
        recos = user_recommendations_dict.get(user_id, [])
        if len(recos) < top_n:
            needed = top_n - len(recos)
            read_set = user_read_articles.get(user_id, set())
            
            # On prend quelques candidats au hasard
            candidates = random.sample(all_articles, needed + 10)
            for c in candidates:
                if c not in recos and c not in read_set:
                    recos.append(c)
                if len(recos) == top_n:
                    break
                    
            user_recommendations_dict[user_id] = recos
            
    print(f"Terminé en {time.time() - start_time:.2f} secondes !")
    return user_recommendations_dict

# Exécution sur l'ensemble de vos données
# user_recos = generate_all_recommendations_optimized(clicks, df_matrix_reco)

# Pour voir un aperçu :
# for user_id in list(user_recos.keys())[:5]:
#     print(f"Utilisateur {user_id}: {user_recos[user_id]}")
"""

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "code_cell_reco_opti",
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_content.split("\n")]
}
code_cell["source"][-1] = code_cell["source"][-1].rstrip("\n")

nb["cells"].extend([md_cell, code_cell])

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Appended optimized cells to notebook.")
