from preprocess import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def find_best_match(query, questions, vectorizer, tfidf_matrix, language='fr', k=1):
    """
    Trouver la question la plus similaire à la requête utilisateur en utilisant KNN.
    Args:
        query (str): Requête utilisateur.
        questions (list): Liste des questions préexistantes.
        vectorizer (TfidfVectorizer): Vectorizer entraîné.
        tfidf_matrix (sparse matrix): Matrice TF-IDF des questions.
        language (str): Langue de la requête ('fr' ou 'en').
        k (int): Nombre de voisins à considérer (par défaut 1).
    Returns:
        tuple: (best_match_idx, similarity_score) - Index de la meilleure correspondance et score.
    """
    # Prétraiter la requête utilisateur
    preprocessed_query = preprocess_text(query, language=language)
    
    # Transformer la requête en vecteur TF-IDF
    query_vector = vectorizer.transform([preprocessed_query])
    
    # Initialiser et ajuster KNN avec la métrique cosinus
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(tfidf_matrix)
    
    # Trouver les k voisins les plus proches
    distances, indices = knn.kneighbors(query_vector)
    
    # L’index de la question la plus proche (premier voisin)
    best_match_idx = indices[0][0]
    # Convertir la distance cosinus en similarité (1 - distance)
    similarity_score = 1 - distances[0][0]
    
    return best_match_idx, similarity_score