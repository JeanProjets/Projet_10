# 📚 GUIDE.md — Construire un Système de Recommandation d'Articles de A à Z

> **Objectif** : Ce guide te conduit pas à pas, de la théorie à l'implémentation, pour construire un système de recommandation qui suggère **5 articles** à un utilisateur donné. Chaque étape inclut des explications pédagogiques et les mathématiques sous-jacentes.

---

## Table des Matières

1. [Étape 0 — Comprendre les Systèmes de Recommandation](#étape-0--comprendre-les-systèmes-de-recommandation)
2. [Étape 1 — Exploration et Compréhension des Données (EDA)](#étape-1--exploration-et-compréhension-des-données-eda)
3. [Étape 2 — Préparation et Nettoyage des Données](#étape-2--préparation-et-nettoyage-des-données)
4. [Étape 3 — Approche 1 : Filtrage par Contenu (Content-Based)](#étape-3--approche-1--filtrage-par-contenu-content-based)
5. [Étape 4 — Approche 2 : Filtrage Collaboratif (Collaborative Filtering)](#étape-4--approche-2--filtrage-collaboratif-collaborative-filtering)
6. [Étape 5 — Approche 3 : Système Hybride](#étape-5--approche-3--système-hybride)
7. [Étape 6 — Gestion du Cold Start](#étape-6--gestion-du-cold-start)
8. [Étape 7 — Évaluation du Système de Recommandation](#étape-7--évaluation-du-système-de-recommandation)
9. [Étape 8 — Réduction de Dimension (ACP/PCA)](#étape-8--réduction-de-dimension-acppca)
10. [Étape 9 — Sérialisation et Export du Modèle](#étape-9--sérialisation-et-export-du-modèle)
11. [Étape 10 — Intégration dans la Cloud Function](#étape-10--intégration-dans-la-cloud-function)
12. [Étape 11 — Architecture Cible (Scalabilité)](#étape-11--architecture-cible-scalabilité)

---

## Étape 0 — Comprendre les Systèmes de Recommandation

### 🎓 Cours : Qu'est-ce qu'un système de recommandation ?

Un **système de recommandation** (ou moteur de recommandation) est un algorithme qui prédit les éléments qu'un utilisateur est susceptible d'apprécier ou de consommer. C'est ce que font Netflix, Spotify, Amazon, YouTube…

Il existe **3 grandes familles** :

#### 1. Filtrage par Contenu (Content-Based Filtering)

**Principe** : On recommande des articles **similaires** à ceux que l'utilisateur a déjà aimés/consultés.

**Analogie** : Imagine que tu lis beaucoup d'articles sur le football. Le système va te recommander d'autres articles sur le football, car leur *contenu* est similaire.

**Comment ça marche** :
1. Chaque article est représenté par un **vecteur de caractéristiques** (embeddings, mots-clés, catégorie…)
2. On construit un **profil utilisateur** = moyenne (ou somme pondérée) des vecteurs des articles qu'il a consultés
3. On calcule la **similarité** entre le profil utilisateur et tous les articles non encore consultés
4. On recommande les articles les plus similaires

**Avantages** :
- Pas besoin de données sur d'autres utilisateurs
- Fonctionne bien même avec peu d'utilisateurs (pas de cold start *utilisateur* au sens strict)
- Explicable : "On te recommande cet article car il est similaire à X que tu as lu"

**Inconvénients** :
- Tendance à la "bulle de filtre" (ne recommande que des choses similaires)
- Ne capte pas les préférences latentes ou surprenantes
- Nécessite une bonne représentation du contenu

#### 2. Filtrage Collaboratif (Collaborative Filtering)

**Principe** : On recommande des articles aimés par **des utilisateurs similaires** à toi.

**Analogie** : Toi et Marie avez lu les mêmes 10 articles. Marie a aussi lu un 11ᵉ article que tu n'as pas lu. Le système te le recommande car "les gens qui aiment les mêmes choses que toi aiment aussi ça".

Il existe deux sous-familles :

##### a) User-Based Collaborative Filtering
- On trouve les utilisateurs **les plus similaires** à l'utilisateur cible
- On recommande les articles que ces utilisateurs similaires ont consultés mais pas l'utilisateur cible

##### b) Item-Based Collaborative Filtering
- On calcule la **similarité entre articles** basée sur les patterns de co-consultation
- "Les utilisateurs qui ont lu l'article A ont aussi lu l'article B"

##### c) Factorisation de Matrices (Matrix Factorization)
- On décompose la matrice utilisateur-article en deux matrices de faible rang
- Approche plus moderne et performante (cf. SVD, ALS…)

**Avantages** :
- Peut découvrir des intérêts "surprenants" (sérendipité)
- Ne nécessite pas de connaître le contenu des articles
- Effet de réseau : plus il y a d'utilisateurs, mieux ça marche

**Inconvénients** :
- **Cold start** : ne fonctionne pas pour un nouvel utilisateur sans historique
- Nécessite beaucoup de données d'interaction
- Problème de scalabilité (matrice très grande et sparse)

#### 3. Système Hybride

**Principe** : Combine Content-Based et Collaborative Filtering pour profiter des avantages des deux.

**Stratégies de combinaison** :
- **Weighted** : Score final = α × score_content + (1-α) × score_collab
- **Switching** : Utilise Content-Based pour les nouveaux utilisateurs, puis bascule vers Collaboratif
- **Cascade** : Un premier modèle filtre, le second affine
- **Feature augmentation** : Les résultats d'un modèle servent de features pour l'autre

---

## Étape 1 — Exploration et Compréhension des Données (EDA)

### 🎓 Cours : Pourquoi l'EDA est cruciale

L'EDA (Exploratory Data Analysis) est la **première étape indispensable** de tout projet Data Science. Elle permet de :
- Comprendre la structure et la qualité des données
- Identifier les anomalies, valeurs manquantes, outliers
- Formuler des hypothèses
- Orienter les choix de modélisation

### 📝 Sous-étapes à réaliser

#### 1.1 Charger et inspecter les données

```python
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
clicks = pd.read_csv('news-portal-user-interactions-by-globocom/clicks_sample.csv')
metadata = pd.read_csv('news-portal-user-interactions-by-globocom/articles_metadata.csv')

with open('news-portal-user-interactions-by-globocom/articles_embeddings.pickle', 'rb') as f:
    embeddings = pickle.load(f)

# Inspection rapide
print(clicks.shape)      # (lignes, colonnes)
print(clicks.dtypes)     # types des colonnes
print(clicks.head())     # premières lignes
print(clicks.describe()) # statistiques descriptives

print(metadata.shape)
print(metadata.dtypes)
print(metadata.head())

print(type(embeddings))  # Probablement un DataFrame ou dict
print(len(embeddings))   # Nombre d'articles avec embeddings
```

#### 1.2 Analyser les distributions

```python
# Nombre de clics par utilisateur
clicks_per_user = clicks.groupby('user_id').size()
print(f"Moyenne clics/user : {clicks_per_user.mean():.2f}")
print(f"Médiane clics/user : {clicks_per_user.median():.2f}")
print(f"Max clics/user : {clicks_per_user.max()}")

# Distribution
clicks_per_user.hist(bins=50)
plt.title("Distribution du nombre de clics par utilisateur")
plt.xlabel("Nombre de clics")
plt.ylabel("Nombre d'utilisateurs")
plt.show()

# Nombre de clics par article
clicks_per_article = clicks.groupby('click_article_id').size()
clicks_per_article.hist(bins=50)
plt.title("Distribution du nombre de clics par article")
plt.show()
```

#### 1.3 Analyser la couverture et la sparsité

```python
n_users = clicks['user_id'].nunique()
n_articles = metadata['article_id'].nunique()
n_interactions = len(clicks)

sparsity = 1 - (n_interactions / (n_users * n_articles))
print(f"Nombre d'utilisateurs : {n_users}")
print(f"Nombre d'articles : {n_articles}")
print(f"Nombre d'interactions : {n_interactions}")
print(f"Sparsité de la matrice : {sparsity:.4%}")
```

### 🎓 Cours : La sparsité des matrices d'interaction

La **sparsité** est un concept fondamental en systèmes de recommandation.

**Définition mathématique** :

$$\text{Sparsité} = 1 - \frac{\text{nombre d'interactions observées}}{\text{nombre d'utilisateurs} \times \text{nombre d'articles}}$$

En pratique, cette matrice est **extrêmement sparse** (souvent > 99%). Un utilisateur typique n'a consulté qu'une infime fraction des articles disponibles.

**Implications** :
- Le filtrage collaboratif classique (user-based, item-based) peut souffrir car il est difficile de trouver des paires d'utilisateurs/articles ayant assez de co-occurrences
- Les approches par factorisation de matrices sont plus robustes face à la sparsité
- Le content-based ne souffre pas de la sparsité (il utilise les features des items, pas la matrice)

#### 1.4 Analyser les catégories d'articles

```python
# Distribution des catégories
category_counts = metadata['category_id'].value_counts()
category_counts.plot(kind='bar', figsize=(12, 5))
plt.title("Distribution des articles par catégorie")
plt.show()

# Catégories les plus cliquées
clicks_with_meta = clicks.merge(metadata, left_on='click_article_id', right_on='article_id')
category_clicks = clicks_with_meta['category_id'].value_counts()
category_clicks.plot(kind='bar', figsize=(12, 5))
plt.title("Distribution des clics par catégorie")
plt.show()
```

#### 1.5 Analyser les embeddings

```python
# Structure des embeddings
if isinstance(embeddings, pd.DataFrame):
    print(f"Shape des embeddings : {embeddings.shape}")
    print(f"Dimension des embeddings : {embeddings.shape[1]}")
elif isinstance(embeddings, dict):
    first_key = list(embeddings.keys())[0]
    print(f"Nombre d'articles : {len(embeddings)}")
    print(f"Dimension d'un embedding : {len(embeddings[first_key])}")
```

---

## Étape 2 — Préparation et Nettoyage des Données

### 📝 Sous-étapes à réaliser

#### 2.1 Gestion des valeurs manquantes

```python
# Vérifier les NaN
print(clicks.isnull().sum())
print(metadata.isnull().sum())

# Traiter si nécessaire (drop ou imputation)
clicks = clicks.dropna(subset=['user_id', 'click_article_id'])
```

#### 2.2 Conversion des types et des timestamps

```python
# Convertir les timestamps en datetime
clicks['click_datetime'] = pd.to_datetime(clicks['click_timestamp'], unit='ms')
clicks['session_start_dt'] = pd.to_datetime(clicks['session_start'], unit='ms')
metadata['created_at_dt'] = pd.to_datetime(metadata['created_at_ts'], unit='ms')
```

#### 2.3 Créer la matrice d'interaction utilisateur-article

```python
# Matrice binaire : 1 si l'utilisateur a cliqué sur l'article, 0 sinon
interaction_matrix = clicks.groupby(['user_id', 'click_article_id']).size().unstack(fill_value=0)

# Binariser (1 = au moins un clic)
interaction_matrix = (interaction_matrix > 0).astype(int)

print(f"Shape : {interaction_matrix.shape}")
# (nombre_users, nombre_articles)
```

### 🎓 Cours : La matrice d'interaction

La **matrice d'interaction** (ou matrice de rating) est au cœur des systèmes de recommandation.

Notation : **R** ∈ ℝ^{m×n} où :
- m = nombre d'utilisateurs
- n = nombre d'articles
- R_{ij} = interaction de l'utilisateur i avec l'article j

Dans notre cas, les interactions sont **implicites** (clics) et non explicites (notes). On travaille donc avec :
- R_{ij} = 1 si l'utilisateur i a cliqué sur l'article j
- R_{ij} = 0 sinon (mais 0 ne signifie pas "n'aime pas", juste "n'a pas vu/cliqué")

**Feedback implicite vs explicite** :
- **Explicite** : Notes (1-5 étoiles), likes/dislikes → l'utilisateur exprime directement sa préférence
- **Implicite** : Clics, temps passé, achats → on déduit la préférence du comportement

Le feedback implicite est plus abondant mais plus bruité. Un clic ne signifie pas forcément que l'utilisateur a aimé l'article.

#### 2.4 Split train/test

```python
from sklearn.model_selection import train_test_split

# Pour l'évaluation, on peut masquer certaines interactions
# Stratégie : pour chaque utilisateur, garder les N dernières interactions pour le test

def train_test_split_temporal(clicks_df, test_ratio=0.2):
    """Split temporel : les dernières interactions de chaque user vont dans le test set."""
    clicks_sorted = clicks_df.sort_values('click_timestamp')
    
    train_list = []
    test_list = []
    
    for user_id, user_clicks in clicks_sorted.groupby('user_id'):
        n = len(user_clicks)
        n_test = max(1, int(n * test_ratio))
        
        train_list.append(user_clicks.iloc[:-n_test])
        test_list.append(user_clicks.iloc[-n_test:])
    
    return pd.concat(train_list), pd.concat(test_list)

train_clicks, test_clicks = train_test_split_temporal(clicks)
print(f"Train : {len(train_clicks)} interactions")
print(f"Test : {len(test_clicks)} interactions")
```

### 🎓 Cours : Le split temporel

En systèmes de recommandation, on ne fait **jamais** un split aléatoire classique. Pourquoi ?

Parce que le système doit prédire le **futur** à partir du **passé**. Si on mélange aléatoirement, on aurait des données futures dans le train set, ce qui est du **data leakage** (fuite de données temporelles).

**Split temporel** : On utilise les interactions passées pour entraîner et les interactions futures pour tester. C'est la méthode la plus réaliste.

---

## Étape 3 — Approche 1 : Filtrage par Contenu (Content-Based)

### 🎓 Cours : Les Embeddings

Un **embedding** est une représentation vectorielle dense d'un objet (mot, phrase, article, image…) dans un espace de dimension réduite.

**Propriété fondamentale** : Les objets sémantiquement proches ont des embeddings proches dans l'espace vectoriel.

Dans notre dataset, chaque article a déjà un embedding pré-calculé. C'est un vecteur de dimension d (par exemple 250) qui capture le "sens" de l'article.

### 🎓 Cours : La Similarité Cosinus

La **similarité cosinus** est la mesure de similarité la plus utilisée pour comparer des embeddings.

**Formule mathématique** :

$$\text{cos}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{||\vec{a}|| \times ||\vec{b}||} = \frac{\sum_{i=1}^{d} a_i \times b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \times \sqrt{\sum_{i=1}^{d} b_i^2}}$$

**Interprétation** :
- cos = 1 → les vecteurs pointent dans la même direction (très similaires)
- cos = 0 → les vecteurs sont orthogonaux (pas de relation)
- cos = -1 → les vecteurs pointent dans des directions opposées

**Pourquoi le cosinus plutôt que la distance euclidienne ?**
- La similarité cosinus mesure l'**angle** entre les vecteurs, pas la distance
- Elle est **invariante à la norme** (la longueur du vecteur n'importe pas)
- C'est important car certains articles peuvent avoir des embeddings de magnitudes différentes sans que ça reflète une différence de contenu

### 📝 Sous-étapes à réaliser

#### 3.1 Construire le profil utilisateur

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_user_profile(user_id, clicks_df, embeddings_df):
    """
    Construit le profil d'un utilisateur = moyenne des embeddings
    des articles qu'il a consultés.
    """
    # Articles consultés par l'utilisateur
    user_articles = clicks_df[clicks_df['user_id'] == user_id]['click_article_id'].unique()
    
    # Filtrer les articles qui ont un embedding
    valid_articles = [a for a in user_articles if a in embeddings_df.index]
    
    if len(valid_articles) == 0:
        return None
    
    # Profil = moyenne des embeddings
    user_embedding = embeddings_df.loc[valid_articles].mean(axis=0).values.reshape(1, -1)
    
    return user_embedding
```

### 🎓 Cours : Le profil utilisateur par moyenne d'embeddings

Le **profil utilisateur** dans une approche content-based est un vecteur qui résume les goûts de l'utilisateur.

**Construction par moyenne** :

$$\vec{u} = \frac{1}{|I_u|} \sum_{i \in I_u} \vec{e}_i$$

Où :
- $\vec{u}$ = vecteur profil de l'utilisateur
- $I_u$ = ensemble des articles consultés par l'utilisateur
- $\vec{e}_i$ = embedding de l'article $i$
- $|I_u|$ = nombre d'articles consultés

**Variante pondérée par le temps** (plus sophistiquée) :

$$\vec{u} = \frac{\sum_{i \in I_u} w_i \cdot \vec{e}_i}{\sum_{i \in I_u} w_i}$$

Où $w_i$ est un poids qui décroît avec le temps (les articles récents comptent plus). Par exemple :

$$w_i = e^{-\lambda \cdot \Delta t_i}$$

avec $\Delta t_i$ = temps écoulé depuis le clic sur l'article $i$, et $\lambda$ un paramètre de décroissance.

#### 3.2 Recommander par similarité cosinus

```python
def recommend_content_based(user_id, clicks_df, embeddings_df, metadata_df, n_recommendations=5):
    """
    Recommande les N articles les plus similaires au profil utilisateur.
    """
    # Construire le profil utilisateur
    user_profile = build_user_profile(user_id, clicks_df, embeddings_df)
    
    if user_profile is None:
        return []
    
    # Articles déjà consultés (à exclure des recommandations)
    already_read = set(clicks_df[clicks_df['user_id'] == user_id]['click_article_id'].unique())
    
    # Calculer la similarité avec TOUS les articles
    all_embeddings = embeddings_df.values
    similarities = cosine_similarity(user_profile, all_embeddings)[0]
    
    # Créer un DataFrame avec les scores
    scores = pd.DataFrame({
        'article_id': embeddings_df.index,
        'similarity_score': similarities
    })
    
    # Exclure les articles déjà lus
    scores = scores[~scores['article_id'].isin(already_read)]
    
    # Trier par score décroissant et prendre les top N
    top_n = scores.nlargest(n_recommendations, 'similarity_score')
    
    return top_n
```

#### 3.3 Amélioration : Pondération par popularité

```python
def recommend_content_based_popularity(user_id, clicks_df, embeddings_df, 
                                        metadata_df, alpha=0.8, n_recommendations=5):
    """
    Combine similarité cosinus et popularité de l'article.
    Score final = alpha * similarité + (1-alpha) * popularité_normalisée
    """
    # Similarité content-based
    content_scores = recommend_content_based(user_id, clicks_df, embeddings_df, 
                                              metadata_df, n_recommendations=100)
    
    # Popularité (nombre total de clics par article, normalisé)
    article_popularity = clicks_df['click_article_id'].value_counts()
    article_popularity_norm = (article_popularity - article_popularity.min()) / \
                               (article_popularity.max() - article_popularity.min())
    
    # Combiner les scores
    content_scores['popularity'] = content_scores['article_id'].map(article_popularity_norm).fillna(0)
    content_scores['final_score'] = alpha * content_scores['similarity_score'] + \
                                     (1 - alpha) * content_scores['popularity']
    
    return content_scores.nlargest(n_recommendations, 'final_score')
```

---

## Étape 4 — Approche 2 : Filtrage Collaboratif (Collaborative Filtering)

### 🎓 Cours : Factorisation de Matrices — SVD (Singular Value Decomposition)

La **SVD** (Décomposition en Valeurs Singulières) est l'algorithme fondamental de la factorisation de matrices en systèmes de recommandation. C'est l'approche qui a rendu célèbre le Netflix Prize en 2006.

**Idée intuitive** : On suppose que les goûts des utilisateurs et les caractéristiques des articles peuvent être expliqués par un petit nombre de **facteurs latents** (des dimensions cachées).

Par exemple, pour des films :
- Facteur 1 pourrait capturer "action vs romance"
- Facteur 2 pourrait capturer "blockbuster vs film d'auteur"
- etc.

**Formulation mathématique** :

On décompose la matrice R en trois matrices :

$$R \approx U \Sigma V^T$$

Où :
- **R** ∈ ℝ^{m×n} : matrice d'interaction (m utilisateurs, n articles)
- **U** ∈ ℝ^{m×k} : matrice des facteurs utilisateurs (chaque ligne = un utilisateur dans l'espace latent)
- **Σ** ∈ ℝ^{k×k} : matrice diagonale des valeurs singulières (importance de chaque facteur)
- **V^T** ∈ ℝ^{k×n} : matrice des facteurs articles (chaque colonne = un article dans l'espace latent)
- **k** : nombre de facteurs latents (hyperparamètre, typiquement 10-100)

**Prédiction** : Le score prédit pour l'utilisateur $u$ et l'article $i$ est :

$$\hat{r}_{ui} = \vec{u}_u \cdot \vec{v}_i = \sum_{f=1}^{k} u_{uf} \cdot \sigma_f \cdot v_{if}$$

### 🎓 Cours : SVD Tronquée pour le Feedback Implicite

Pour le **feedback implicite** (clics, pas de notes), on utilise une variante appelée **SVD tronquée** ou **ALS implicite** (Alternating Least Squares).

La différence clé : on ne cherche pas à prédire le rating, mais la **confiance** dans une préférence.

**Modèle de Hu, Koren & Volinsky (2008)** :

On définit :
- $p_{ui}$ = préférence binaire (1 si l'utilisateur a interagi, 0 sinon)
- $c_{ui}$ = confiance = 1 + α × r_{ui} (plus d'interactions = plus de confiance)

On minimise :

$$\min_{U, V} \sum_{u,i} c_{ui} \left( p_{ui} - \vec{u}_u^T \vec{v}_i \right)^2 + \lambda \left( ||\vec{u}_u||^2 + ||\vec{v}_i||^2 \right)$$

Le terme $\lambda(||u||^2 + ||v||^2)$ est la **régularisation L2** qui empêche le surapprentissage.

### 📝 Sous-étapes à réaliser

#### 4.1 SVD tronquée avec scikit-learn

```python
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Créer la matrice sparse
user_ids = train_clicks['user_id'].unique()
article_ids = train_clicks['click_article_id'].unique()

user_idx = {uid: i for i, uid in enumerate(user_ids)}
article_idx = {aid: i for i, aid in enumerate(article_ids)}

rows = train_clicks['user_id'].map(user_idx)
cols = train_clicks['click_article_id'].map(article_idx)
data = np.ones(len(train_clicks))

interaction_sparse = csr_matrix((data, (rows, cols)), 
                                 shape=(len(user_ids), len(article_ids)))

# SVD tronquée
n_components = 50  # nombre de facteurs latents
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(interaction_sparse)  # U * Sigma
item_factors = svd.components_.T  # V

print(f"Variance expliquée : {svd.explained_variance_ratio_.sum():.4f}")
print(f"User factors shape : {user_factors.shape}")
print(f"Item factors shape : {item_factors.shape}")
```

### 🎓 Cours : La Variance Expliquée

La **variance expliquée** mesure la proportion d'information capturée par les k premiers facteurs latents.

$$\text{Variance expliquée} = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{n} \sigma_i^2}$$

Où $\sigma_i$ sont les valeurs singulières ordonnées par décroissance.

Si la variance expliquée avec k=50 est de 0.30 (30%), cela signifie que 50 facteurs capturent 30% de la structure des données. C'est normal pour des données très sparse — même un faible pourcentage peut produire de bonnes recommandations.

#### 4.2 Recommander avec SVD

```python
def recommend_svd(user_id, user_factors, item_factors, 
                   user_idx, article_idx, clicks_df, n_recommendations=5):
    """
    Recommande des articles via factorisation de matrices (SVD).
    """
    if user_id not in user_idx:
        return []
    
    u_idx = user_idx[user_id]
    user_vec = user_factors[u_idx]  # Vecteur de l'utilisateur dans l'espace latent
    
    # Calculer les scores pour tous les articles
    scores = user_vec @ item_factors.T  # produit scalaire
    
    # Articles déjà lus
    already_read = set(clicks_df[clicks_df['user_id'] == user_id]['click_article_id'].unique())
    
    # Créer le mapping inverse
    idx_to_article = {v: k for k, v in article_idx.items()}
    
    # Trier et filtrer
    article_scores = [(idx_to_article[i], scores[i]) for i in range(len(scores))
                       if idx_to_article[i] not in already_read]
    
    article_scores.sort(key=lambda x: x[1], reverse=True)
    
    return article_scores[:n_recommendations]
```

#### 4.3 Alternative : User-Based KNN Collaborative Filtering

```python
from sklearn.neighbors import NearestNeighbors

def recommend_user_knn(user_id, interaction_sparse, user_idx, article_idx,
                        clicks_df, k=20, n_recommendations=5):
    """
    Recommande en trouvant les K utilisateurs les plus similaires (KNN)
    et en agrégeant leurs articles lus.
    """
    if user_id not in user_idx:
        return []
    
    # Fit KNN sur la matrice d'interaction
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
    knn.fit(interaction_sparse)
    
    u_idx = user_idx[user_id]
    distances, indices = knn.kneighbors(interaction_sparse[u_idx])
    
    # Exclure l'utilisateur lui-même
    neighbor_indices = indices[0][1:]
    neighbor_distances = distances[0][1:]
    
    # Agréger les articles des voisins (pondérés par la similarité)
    idx_to_article = {v: k for k, v in article_idx.items()}
    already_read = set(clicks_df[clicks_df['user_id'] == user_id]['click_article_id'].unique())
    
    article_scores = {}
    for n_idx, dist in zip(neighbor_indices, neighbor_distances):
        similarity = 1 - dist  # cosine similarity = 1 - cosine distance
        neighbor_articles = interaction_sparse[n_idx].nonzero()[1]
        
        for art_idx in neighbor_articles:
            art_id = idx_to_article[art_idx]
            if art_id not in already_read:
                article_scores[art_id] = article_scores.get(art_id, 0) + similarity
    
    # Trier par score
    sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_articles[:n_recommendations]
```

### 🎓 Cours : K-Nearest Neighbors (KNN) en Recommandation

Le **KNN collaboratif** est l'approche la plus intuitive :

1. **Trouver les K voisins** : Pour un utilisateur u, on cherche les K utilisateurs les plus similaires en utilisant la similarité cosinus sur les vecteurs d'interaction
2. **Agréger** : On collecte les articles lus par ces voisins mais pas par u
3. **Pondérer** : Les articles recommandés par des voisins plus proches ont un poids plus fort
4. **Trier** : On retourne les articles avec le score le plus élevé

**Calcul de la similarité entre utilisateurs** :

$$\text{sim}(u, v) = \frac{\vec{r}_u \cdot \vec{r}_v}{||\vec{r}_u|| \times ||\vec{r}_v||}$$

Où $\vec{r}_u$ est la ligne de la matrice d'interaction pour l'utilisateur u.

**Score de prédiction** :

$$\hat{r}_{ui} = \frac{\sum_{v \in N_k(u)} \text{sim}(u, v) \cdot r_{vi}}{\sum_{v \in N_k(u)} |\text{sim}(u, v)|}$$

---

## Étape 5 — Approche 3 : Système Hybride

### 🎓 Cours : Pourquoi l'hybride ?

L'approche hybride est généralement **la plus performante** car elle combine les forces des deux approches :

| Critère | Content-Based | Collaboratif | Hybride |
|---|---|---|---|
| Nouvel utilisateur | ✅ (si ≥1 interaction) | ❌ (cold start) | ✅ |
| Nouvel article | ✅ (si embedding dispo) | ❌ (cold start) | ✅ |
| Sérendipité | ❌ (bulle de filtre) | ✅ | ✅ |
| Données nécessaires | Embeddings | Interactions | Les deux |

### 📝 Sous-étapes à réaliser

#### 5.1 Hybride par pondération linéaire

```python
def recommend_hybrid(user_id, clicks_df, embeddings_df, metadata_df,
                      user_factors, item_factors, user_idx, article_idx,
                      alpha=0.5, n_recommendations=5):
    """
    Système hybride : combine content-based et collaborative filtering.
    Score final = alpha * score_content + (1-alpha) * score_collab
    """
    # --- Score Content-Based ---
    user_profile = build_user_profile(user_id, clicks_df, embeddings_df)
    already_read = set(clicks_df[clicks_df['user_id'] == user_id]['click_article_id'].unique())
    
    if user_profile is not None:
        all_similarities = cosine_similarity(user_profile, embeddings_df.values)[0]
        content_scores = pd.Series(all_similarities, index=embeddings_df.index)
    else:
        content_scores = pd.Series(0, index=embeddings_df.index)
    
    # Normaliser entre 0 et 1
    if content_scores.max() > content_scores.min():
        content_scores = (content_scores - content_scores.min()) / \
                          (content_scores.max() - content_scores.min())
    
    # --- Score Collaboratif (SVD) ---
    if user_id in user_idx:
        u_idx = user_idx[user_id]
        user_vec = user_factors[u_idx]
        svd_scores_raw = user_vec @ item_factors.T
        
        idx_to_article = {v: k for k, v in article_idx.items()}
        collab_scores = pd.Series(0.0, index=embeddings_df.index)
        for i, score in enumerate(svd_scores_raw):
            art_id = idx_to_article.get(i)
            if art_id is not None and art_id in collab_scores.index:
                collab_scores[art_id] = score
    else:
        collab_scores = pd.Series(0, index=embeddings_df.index)
    
    # Normaliser entre 0 et 1
    if collab_scores.max() > collab_scores.min():
        collab_scores = (collab_scores - collab_scores.min()) / \
                          (collab_scores.max() - collab_scores.min())
    
    # --- Score Hybride ---
    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores
    
    # Exclure les articles déjà lus
    hybrid_scores = hybrid_scores.drop(index=list(already_read), errors='ignore')
    
    # Top N
    top_articles = hybrid_scores.nlargest(n_recommendations)
    
    return pd.DataFrame({
        'article_id': top_articles.index,
        'hybrid_score': top_articles.values,
        'content_score': content_scores.reindex(top_articles.index).values,
        'collab_score': collab_scores.reindex(top_articles.index).values
    })
```

#### 5.2 Optimisation de alpha par validation

```python
def optimize_alpha(train_clicks, test_clicks, embeddings_df, metadata_df,
                    user_factors, item_factors, user_idx, article_idx,
                    alpha_range=np.arange(0.0, 1.05, 0.1)):
    """
    Trouve le meilleur alpha par validation sur le test set.
    """
    results = []
    
    for alpha in alpha_range:
        hits = 0
        total = 0
        
        for user_id in test_clicks['user_id'].unique():
            # Articles réellement consultés dans le test
            true_articles = set(test_clicks[test_clicks['user_id'] == user_id]['click_article_id'])
            
            # Recommandations
            recs = recommend_hybrid(user_id, train_clicks, embeddings_df, metadata_df,
                                     user_factors, item_factors, user_idx, article_idx,
                                     alpha=alpha, n_recommendations=5)
            
            if len(recs) > 0:
                rec_articles = set(recs['article_id'])
                hits += len(true_articles & rec_articles)
                total += len(true_articles)
        
        hit_rate = hits / total if total > 0 else 0
        results.append({'alpha': alpha, 'hit_rate': hit_rate})
        print(f"alpha={alpha:.1f} → Hit Rate = {hit_rate:.4f}")
    
    return pd.DataFrame(results)
```

---

## Étape 6 — Gestion du Cold Start

### 🎓 Cours : Le problème du Cold Start

Le **cold start** est le problème majeur des systèmes de recommandation. Il se manifeste dans deux cas :

#### Cold Start — Nouvel Utilisateur
Un utilisateur vient de s'inscrire, on n'a aucune interaction. Comment lui recommander des articles ?

**Solutions** :
1. **Recommandation populaire** : Recommander les articles les plus populaires (baseline simple mais efficace)
2. **Onboarding** : Demander à l'utilisateur de sélectionner des catégories qui l'intéressent
3. **Données démographiques** : Si on a l'âge, le pays, la langue → recommander ce qui est populaire dans son segment
4. **Articles récents** : Recommander les articles les plus récents et les plus populaires

#### Cold Start — Nouvel Article
Un nouvel article vient d'être publié, personne ne l'a encore lu. Comment le recommander ?

**Solutions** :
1. **Content-Based** : On peut calculer son embedding et le recommander aux utilisateurs dont le profil est similaire (c'est l'avantage du content-based !)
2. **Métadonnées** : Utiliser sa catégorie, son publisher, son nombre de mots pour le rapprocher d'articles existants
3. **Exploration** : Injecter volontairement de nouveaux articles dans les recommandations (stratégie "explore vs exploit")

### 📝 Sous-étapes à réaliser

#### 6.1 Fonction de recommandation avec fallback

```python
def recommend_with_cold_start(user_id, clicks_df, embeddings_df, metadata_df,
                               user_factors, item_factors, user_idx, article_idx,
                               n_recommendations=5):
    """
    Système de recommandation robuste avec gestion du cold start.
    """
    user_clicks = clicks_df[clicks_df['user_id'] == user_id]
    n_interactions = len(user_clicks)
    
    if n_interactions == 0:
        # COLD START TOTAL → Articles populaires récents
        print(f"[Cold Start] User {user_id} : aucune interaction → articles populaires")
        return recommend_popular(clicks_df, metadata_df, n_recommendations)
    
    elif n_interactions < 5:
        # COLD START PARTIEL → Content-Based uniquement (peu de données pour le collab)
        print(f"[Warm Start] User {user_id} : {n_interactions} interactions → content-based")
        return recommend_content_based(user_id, clicks_df, embeddings_df, 
                                        metadata_df, n_recommendations)
    
    else:
        # UTILISATEUR ACTIF → Hybride
        print(f"[Active User] User {user_id} : {n_interactions} interactions → hybride")
        return recommend_hybrid(user_id, clicks_df, embeddings_df, metadata_df,
                                 user_factors, item_factors, user_idx, article_idx,
                                 alpha=0.6, n_recommendations=n_recommendations)


def recommend_popular(clicks_df, metadata_df, n_recommendations=5):
    """
    Recommande les articles les plus populaires (fallback cold start).
    """
    popular = clicks_df['click_article_id'].value_counts().head(n_recommendations)
    
    return pd.DataFrame({
        'article_id': popular.index,
        'score': popular.values / popular.values.max()  # Normaliser
    })
```

---

## Étape 7 — Évaluation du Système de Recommandation

### 🎓 Cours : Métriques d'Évaluation

Évaluer un système de recommandation est différent d'évaluer un classifieur classique. Voici les métriques standard :

#### 1. Hit Rate (taux de réussite) @ K

**Définition** : Proportion d'utilisateurs pour lesquels au moins un article recommandé est dans le test set.

$$\text{Hit Rate@K} = \frac{|\{u : |R_K(u) \cap T(u)| > 0\}|}{|U|}$$

Où :
- $R_K(u)$ = top K articles recommandés pour l'utilisateur u
- $T(u)$ = articles du test set pour u
- $U$ = ensemble des utilisateurs du test

#### 2. Precision @ K

**Définition** : Proportion d'articles recommandés qui sont pertinents.

$$\text{Precision@K} = \frac{|R_K(u) \cap T(u)|}{K}$$

**Exemple** : Si on recommande 5 articles et que 2 sont dans le test set → Precision@5 = 2/5 = 0.40

#### 3. Recall @ K

**Définition** : Proportion d'articles pertinents qui ont été recommandés.

$$\text{Recall@K} = \frac{|R_K(u) \cap T(u)|}{|T(u)|}$$

**Exemple** : Si le test set contient 3 articles et qu'on en retrouve 2 parmi les 5 recommandés → Recall@5 = 2/3 = 0.67

#### 4. NDCG @ K (Normalized Discounted Cumulative Gain)

**Définition** : Mesure la qualité du **classement** des recommandations (les articles pertinents doivent être en haut de la liste).

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Où :
- $rel_i$ = 1 si l'article en position i est pertinent, 0 sinon
- $\text{IDCG@K}$ = DCG du classement parfait (tous les pertinents en premier)

**Intuition** : Un article pertinent en position 1 contribue plus qu'un article pertinent en position 5. Le $\log_2(i+1)$ au dénominateur pénalise les positions basses.

#### 5. MAP @ K (Mean Average Precision)

$$\text{AP@K} = \frac{1}{\min(m, K)} \sum_{k=1}^{K} P(k) \cdot rel(k)$$

Où $P(k)$ est la precision au rang $k$ et $m$ le nombre total de documents pertinents.

### 📝 Sous-étapes à réaliser

#### 7.1 Implémenter les métriques

```python
def precision_at_k(recommended, actual, k=5):
    """Precision@K"""
    rec_set = set(recommended[:k])
    act_set = set(actual)
    return len(rec_set & act_set) / k

def recall_at_k(recommended, actual, k=5):
    """Recall@K"""
    rec_set = set(recommended[:k])
    act_set = set(actual)
    if len(act_set) == 0:
        return 0
    return len(rec_set & act_set) / len(act_set)

def ndcg_at_k(recommended, actual, k=5):
    """NDCG@K"""
    act_set = set(actual)
    
    # DCG
    dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k])
               if item in act_set])
    
    # Ideal DCG
    n_relevant = min(len(act_set), k)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(n_relevant)])
    
    return dcg / idcg if idcg > 0 else 0

def evaluate_recommender(recommend_fn, test_clicks, k=5, **kwargs):
    """
    Évalue un système de recommandation sur le test set.
    """
    precisions = []
    recalls = []
    ndcgs = []
    hits = 0
    
    users = test_clicks['user_id'].unique()
    
    for user_id in users:
        # Articles réels du test
        actual = test_clicks[test_clicks['user_id'] == user_id]['click_article_id'].tolist()
        
        # Recommandations
        recs = recommend_fn(user_id, **kwargs)
        if isinstance(recs, pd.DataFrame):
            recommended = recs['article_id'].tolist()
        else:
            recommended = [r[0] for r in recs] if recs else []
        
        if len(recommended) == 0:
            continue
        
        p = precision_at_k(recommended, actual, k)
        r = recall_at_k(recommended, actual, k)
        n = ndcg_at_k(recommended, actual, k)
        
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)
        
        if len(set(recommended[:k]) & set(actual)) > 0:
            hits += 1
    
    print(f"=== Résultats @{k} ===")
    print(f"Hit Rate    : {hits / len(users):.4f}")
    print(f"Precision   : {np.mean(precisions):.4f}")
    print(f"Recall      : {np.mean(recalls):.4f}")
    print(f"NDCG        : {np.mean(ndcgs):.4f}")
    
    return {
        'hit_rate': hits / len(users),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'ndcg': np.mean(ndcgs)
    }
```

#### 7.2 Comparer les approches

```python
# Évaluer chaque approche
print("\n--- Content-Based ---")
results_cb = evaluate_recommender(recommend_content_based, test_clicks,
                                   clicks_df=train_clicks, embeddings_df=embeddings_df,
                                   metadata_df=metadata)

print("\n--- Collaborative (SVD) ---")
results_svd = evaluate_recommender(recommend_svd, test_clicks,
                                    user_factors=user_factors, item_factors=item_factors,
                                    user_idx=user_idx, article_idx=article_idx,
                                    clicks_df=train_clicks)

print("\n--- Hybride ---")
results_hybrid = evaluate_recommender(recommend_hybrid, test_clicks,
                                       clicks_df=train_clicks, embeddings_df=embeddings_df,
                                       metadata_df=metadata, user_factors=user_factors,
                                       item_factors=item_factors, user_idx=user_idx,
                                       article_idx=article_idx)

# Tableau comparatif
comparison = pd.DataFrame([results_cb, results_svd, results_hybrid],
                           index=['Content-Based', 'SVD Collab', 'Hybride'])
print("\n=== COMPARAISON ===")
print(comparison.round(4))
```

---

## Étape 8 — Réduction de Dimension (ACP/PCA)

### 🎓 Cours : L'Analyse en Composantes Principales (ACP/PCA)

L'**ACP** (Analyse en Composantes Principales) ou **PCA** (Principal Component Analysis) est une technique de réduction de dimensionnalité.

**Pourquoi en a-t-on besoin ?**
- Les embeddings font 250 dimensions → le fichier fait ~364 Mo
- Les services cloud gratuits ont des limites de taille
- Réduire la dimension accélère les calculs de similarité

**Principe mathématique** :

1. **Centrage** : On soustrait la moyenne de chaque feature :
   $$\tilde{X} = X - \bar{X}$$

2. **Matrice de covariance** :
   $$C = \frac{1}{n-1} \tilde{X}^T \tilde{X}$$

3. **Décomposition en valeurs propres** :
   $$C \vec{v}_k = \lambda_k \vec{v}_k$$
   
   Où $\vec{v}_k$ sont les **vecteurs propres** (directions principales) et $\lambda_k$ les **valeurs propres** (variance dans chaque direction).

4. **Projection** : On projette les données sur les k premiers vecteurs propres (ceux avec les plus grandes valeurs propres) :
   $$X_{réduit} = \tilde{X} \cdot V_k$$

**La variance expliquée** par la k-ème composante :

$$\frac{\lambda_k}{\sum_{i=1}^{d} \lambda_i}$$

On choisit k tel que la **variance expliquée cumulée** soit suffisante (typiquement > 90%).

### 📝 Sous-étapes à réaliser

#### 8.1 Appliquer la PCA aux embeddings

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger les embeddings
embeddings_matrix = embeddings_df.values  # (n_articles, d_original)

# Analyser la variance expliquée pour choisir k
pca_full = PCA(n_components=min(100, embeddings_matrix.shape[1]))
pca_full.fit(embeddings_matrix)

# Tracer la variance expliquée cumulée
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-o', markersize=3)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% de variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% de variance')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée cumulée')
plt.title('Choix du nombre de composantes PCA')
plt.legend()
plt.grid(True)
plt.show()

# Trouver k pour 90% et 95% de variance
k_90 = np.argmax(cumulative_variance >= 0.90) + 1
k_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"k pour 90% de variance : {k_90}")
print(f"k pour 95% de variance : {k_95}")
```

#### 8.2 Réduire les embeddings et vérifier la qualité

```python
# Appliquer la PCA avec le k choisi
k_chosen = k_90  # Par exemple 50
pca = PCA(n_components=k_chosen)
embeddings_reduced = pca.fit_transform(embeddings_matrix)

print(f"Shape originale : {embeddings_matrix.shape}")
print(f"Shape réduite : {embeddings_reduced.shape}")
print(f"Réduction de taille : {embeddings_matrix.nbytes / embeddings_reduced.nbytes:.1f}x")

# Vérifier que les recommandations ne se dégradent pas trop
# (refaire les étapes 3-7 avec les embeddings réduits et comparer)
embeddings_reduced_df = pd.DataFrame(embeddings_reduced, index=embeddings_df.index)
```

#### 8.3 Sauvegarder les embeddings réduits

```python
import pickle

# Sauvegarder les embeddings réduits
with open('models/embeddings_reduced.pickle', 'wb') as f:
    pickle.dump(embeddings_reduced_df, f)

# Sauvegarder le modèle PCA (pour transformer les embeddings de nouveaux articles)
with open('models/pca_model.pickle', 'wb') as f:
    pickle.dump(pca, f)

print(f"Taille du fichier réduit : {embeddings_reduced_df.values.nbytes / 1024 / 1024:.1f} Mo")
```

---

## Étape 9 — Sérialisation et Export du Modèle

### 🎓 Cours : La Sérialisation

**Sérialiser** un modèle, c'est le transformer en fichier binaire qu'on pourra recharger plus tard sans avoir à ré-entraîner.

Formats courants en Python :
- **pickle** : Format natif Python, le plus simple
- **joblib** : Optimisé pour les gros tableaux numpy (plus rapide pour les modèles scikit-learn)
- **ONNX** : Format universel, interopérable entre frameworks

### 📝 Sous-étapes à réaliser

#### 9.1 Sauvegarder tout le nécessaire

```python
import pickle
import joblib

# Créer le dossier models/
import os
os.makedirs('models', exist_ok=True)

# 1. Embeddings réduits (déjà fait ci-dessus)

# 2. Facteurs SVD
model_data = {
    'user_factors': user_factors,
    'item_factors': item_factors,
    'user_idx': user_idx,
    'article_idx': article_idx,
    'n_components': n_components,
}
with open('models/svd_model.pickle', 'wb') as f:
    pickle.dump(model_data, f)

# 3. Données de popularité (pour le cold start)
popular_articles = clicks['click_article_id'].value_counts().head(20).index.tolist()
with open('models/popular_articles.pickle', 'wb') as f:
    pickle.dump(popular_articles, f)

# 4. Métadonnées articles (version légère)
metadata_light = metadata[['article_id', 'category_id', 'words_count']].copy()
metadata_light.to_csv('models/articles_metadata_light.csv', index=False)

print("Tous les artefacts sont sauvegardés dans models/")
```

---

## Étape 10 — Intégration dans la Cloud Function

### 📝 Sous-étapes à réaliser

1. Uploader les modèles dans un **Google Cloud Storage bucket**
2. Écrire la **Cloud Function** qui :
   - Charge les modèles depuis GCS au démarrage
   - Reçoit un `user_id` en entrée (requête HTTP)
   - Retourne les 5 articles recommandés en JSON
3. Déployer avec `gcloud functions deploy`

Le code complet est dans le fichier `tuto_cloud_functions.md`.

---

## Étape 11 — Architecture Cible (Scalabilité)

### 🎓 Cours : Architecture Cible pour la Scalabilité

Pour la présentation, tu dois décrire une **architecture cible** qui gère :

#### Ajout de nouveaux utilisateurs
- **Content-Based** : Dès qu'un nouveau user fait son premier clic, on peut calculer son profil et recommander (pas de ré-entraînement nécessaire)
- **Collaboratif** : Nécessite un ré-entraînement périodique (batch) ou une mise à jour incrémentale (online learning)
- **Cold Start** : Les nouveaux utilisateurs reçoivent des recommandations populaires, puis basculent vers le content-based, puis vers l'hybride

#### Ajout de nouveaux articles
- **Embeddings** : Un pipeline de génération d'embeddings est déclenché à chaque publication (modèle NLP pré-entraîné type BERT/Sentence-Transformers)
- **PCA** : Les nouveaux embeddings sont projetés dans l'espace réduit avec le modèle PCA sauvegardé
- **Indexation** : Le nouvel article est ajouté à l'index de recherche de similarité

#### Architecture cible (schéma conceptuel)

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE DE DONNÉES                   │
│                                                         │
│  [Nouvel Article] → [NLP Embeddings] → [PCA] → [GCS]   │
│  [Nouveau User]   → [Premier clic]   → [Profil User]   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                    BATCH PROCESSING                      │
│                                                         │
│  [Cloud Scheduler] → [Cloud Function: Retrain]          │
│  (chaque nuit)       - SVD sur nouvelles interactions    │
│                      - Mise à jour user/item factors     │
│                      - Sauvegarde dans GCS               │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                    API TEMPS RÉEL                        │
│                                                         │
│  [Streamlit App] → [Cloud Function: Recommend]          │
│                     - Charge modèles depuis GCS         │
│                     - Calcule recommandations            │
│                     - Retourne Top 5 articles            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🏁 Résumé des Étapes

| # | Étape | Livrable | Priorité |
|---|---|---|---|
| 0 | Comprendre la théorie | Notes de compréhension | ⭐⭐⭐ |
| 1 | EDA | Notebook d'exploration | ⭐⭐⭐ |
| 2 | Préparer les données | Matrice d'interaction, split | ⭐⭐⭐ |
| 3 | Content-Based | Fonction de recommandation CB | ⭐⭐⭐ |
| 4 | Collaborative Filtering | SVD + KNN | ⭐⭐⭐ |
| 5 | Système Hybride | Fonction hybride | ⭐⭐ |
| 6 | Cold Start | Fallback strategy | ⭐⭐ |
| 7 | Évaluation | Métriques comparatives | ⭐⭐⭐ |
| 8 | PCA | Embeddings réduits | ⭐⭐ |
| 9 | Sérialisation | Modèles sauvegardés | ⭐⭐ |
| 10 | Cloud Function | Déploiement serverless | ⭐⭐⭐ |
| 11 | Architecture cible | Schéma pour la présentation | ⭐⭐ |

---

> **Conseil** : Travaille étape par étape dans un notebook Jupyter. Chaque étape = une section du notebook. À la fin, tu extrairas le code "propre" dans des scripts Python pour le déploiement.
