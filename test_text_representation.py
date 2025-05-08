from text_representation import create_tfidf_vectors
from chatbot_model import find_best_match

# Exemple de jeu de données (simulant les questions de l’étape 2)
questions = [
    "Quelles sont les conditions d'admission ?",
    "Quelle est la date limite d'inscription ?",
    "Quels sont les cours disponibles à l'ISET ?"
]
answers = [
    "Vous devez avoir un diplôme de baccalauréat...",
    "La date limite est le 30 juillet 2025...",
    "Les cours incluent informatique, gestion..."
]

# Créer les vecteurs TF-IDF
vectorizer, tfidf_matrix = create_tfidf_vectors(questions, language='fr')

# Tester une requête utilisateur
query = "Critères d'admission à l'ISET ?"
best_match_idx, similarity_score = find_best_match(query, questions, vectorizer, tfidf_matrix, language='fr')
print(f"Requête : {query}")
print(f"Meilleure correspondance : {questions[best_match_idx]}")
print(f"Réponse : {answers[best_match_idx]}")
print(f"Score de similarité : {similarity_score:.4f}")