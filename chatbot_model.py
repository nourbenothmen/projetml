from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from text_representation import create_tfidf_vectors, transform_query

def find_best_match(query, questions, vectorizer, tfidf_matrix):
    """
    Trouver la question la plus similaire à la requête utilisateur.
    Args:
        query (str): Requête utilisateur.
        questions (list): Liste des questions du jeu de données.
        vectorizer: Vectoriseur TF-IDF entraîné.
        tfidf_matrix: Matrice TF-IDF des questions.
    Returns:
        tuple: (index de la meilleure correspondance, score de similarité)
    """
    # Convertir la requête en vecteur TF-IDF
    query_vector = transform_query(query, vectorizer)
    # Calculer la similarité cosinus
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    # Trouver l’index de la question la plus similaire
    best_match_idx = np.argmax(similarities)
    return best_match_idx, similarities[0][best_match_idx]