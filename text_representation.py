from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_text

def create_tfidf_vectors(questions):
    """
    Convertir une liste de questions en vecteurs TF-IDF.
    Args:
        questions (list): Liste de questions (textes).
    Returns:
        tuple: (vectorizer, tfidf_matrix)
    """
    # Prétraiter les questions
    processed_questions = [preprocess_text(q) for q in questions]
    # Initialiser le vectoriseur TF-IDF
    vectorizer = TfidfVectorizer()
    # Créer la matrice TF-IDF
    tfidf_matrix = vectorizer.fit_transform(processed_questions)
    return vectorizer, tfidf_matrix

def transform_query(query, vectorizer):
    """
    Convertir une requête utilisateur en vecteur TF-IDF.
    Args:
        query (str): Requête utilisateur.
        vectorizer: Vectoriseur TF-IDF entraîné.
    Returns:
        scipy.sparse.csr_matrix: Vecteur TF-IDF de la requête.
    """
    processed_query = preprocess_text(query)
    return vectorizer.transform([processed_query])