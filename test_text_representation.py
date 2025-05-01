from text_representation import create_tfidf_vectors, transform_query

# Exemple de jeu de données (simulant les questions de l’étape 2)
questions = [
    "Quelles sont les conditions d'admission ?",
    "Quelle est la date limite d'inscription ?",
    "Quels sont les cours disponibles à l'ISET ?"
]

# Créer les vecteurs TF-IDF
vectorizer, tfidf_matrix = create_tfidf_vectors(questions)
print("Matrice TF-IDF des questions :")
print(tfidf_matrix.toarray())

# Tester une requête utilisateur
query = "Conditions d'admission à l'ISET ?"
query_vector = transform_query(query, vectorizer)
print(f"Vecteur TF-IDF de la requête '{query}' :")
print(query_vector.toarray())