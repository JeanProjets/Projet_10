import nbformat

# Read the notebook
nb = nbformat.read("notebooks/notebook_test.ipynb", as_version=4)

# Create markdown cell explaining the approach
md_content = """## Étape suivante : Recommandation pour les utilisateurs (Version Simple)

Voici une implémentation simple de votre idée :
1. On charge l'historique des clics (ex: `clicks_sample.csv`).
2. Pour chaque utilisateur, on récupère ses 5 derniers articles lus.
3. Pour chacun de ces articles, on récupère les 5 articles similaires depuis notre matrice `df_matrice_recommandation`.
4. On compte les occurrences de chaque article recommandé (les "overlaps"). Les articles recommandés plusieurs fois remontent dans le classement.
5. On complète avec des articles au hasard si on n'atteint pas 5 recommandations uniques."""

md_cell = nbformat.v4.new_markdown_cell(md_content)

# Create code cell
code_content = """import pandas as pd
from collections import Counter
import random

# 1. Charger les données d'interactions (on teste avec le sample d'abord)
clicks_sample = pd.read_csv("../news-portal-user-interactions-by-globocom/clicks_sample.csv")

# 2. Charger la matrice de recommandations sauvegardée précédemment 
# (si df_matrice_recommandation est déjà en mémoire, vous pouvez utiliser df_matrice_recommandation directement)
df_matrix_reco = pd.read_csv("matrix_recommendation_article_for_content_based.csv", index_col=0)
df_matrix_reco.columns = ['self', 'reco_1', 'reco_2', 'reco_3', 'reco_4', 'reco_5']

def get_user_recommendation(user_id, clicks_df, matrix_reco_df, top_n=5):
    # a. Historique de l'utilisateur (les plus récents en premier)
    user_clicks = clicks_df[clicks_df['user_id'] == user_id].sort_values('click_timestamp', ascending=False)
    
    # b. Prendre les 5 derniers articles lus
    last_5_articles = user_clicks['click_article_id'].head(5).tolist()
    
    if not last_5_articles:
        # S'il n'a rien lu, on recommande 5 articles au hasard
        return random.sample(matrix_reco_df.index.tolist(), top_n)
    
    # c. Pour chaque article lu, récupérer ses 5 recommandations
    all_recommendations = []
    for article_id in last_5_articles:
        if article_id in matrix_reco_df.index:
            # On prend les recos 1 à 5 (on ignore l'article lui-même 'self')
            recos = matrix_reco_df.loc[article_id, ['reco_1', 'reco_2', 'reco_3', 'reco_4', 'reco_5']].tolist()
            all_recommendations.extend(recos)
            
    # d. Compter les occurrences (plus il y a de recommandations du même article, mieux c'est)
    reco_counts = Counter(all_recommendations)
    
    # On retire les articles que l'utilisateur a DÉJÀ lu de la liste des recommandations (optionnel mais recommandé)
    for article in last_5_articles:
        if article in reco_counts:
            del reco_counts[article]
            
    # e. Trier par nombre d'occurrences (most_common trie automatiquement par valeur décroissante)
    sorted_recos = [article for article, count in reco_counts.most_common(top_n)]
    
    # f. Si on n'a pas 5 recommandations uniques, on complète au hasard
    if len(sorted_recos) < top_n:
        needed = top_n - len(sorted_recos)
        random_candidates = random.sample(matrix_reco_df.index.tolist(), needed + 5) # on en prend un peu plus au cas où
        for article in random_candidates:
            if article not in sorted_recos and article not in last_5_articles:
                sorted_recos.append(article)
            if len(sorted_recos) == top_n:
                break
                
    return sorted_recos

# 3. Tester sur quelques utilisateurs du sample
test_users = clicks_sample['user_id'].unique()[:5]

for user in test_users:
    recos = get_user_recommendation(user, clicks_sample, df_matrix_reco)
    print(f"Recommandations pour l'utilisateur {user} : {recos}")
"""

code_cell = nbformat.v4.new_code_cell(code_content)

# Add the new cells to the notebook
nb.cells.extend([md_cell, code_cell])

# Write back to the file
nbformat.write(nb, "notebooks/notebook_test.ipynb")
print("Successfully appended cells to notebook_test.ipynb")
