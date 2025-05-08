from app import create_app
from text_representation import preprocess_text

app = create_app()

with app.app_context():
    test_questions = [
        "Quels sont les cours de Génie Civil?",
        "What are the Computer Science courses?",
        "Comment s'inscrire à l'ISET?",
        "Quelles sont les dates d'inscription?"
    ]

    for query in test_questions:
        try:
            language = 'fr' if query.startswith('Qu') else 'en'
            processed_query = preprocess_text(query, language=language)
            print(f"Question: {query}")
            print(f"Query prétraitée: {processed_query}")

            predicted_category = None
            if app.category_classifier and app.category_vectorizer:
                query_vector = app.category_vectorizer.transform([processed_query])
                predicted_category = app.category_classifier.predict(query_vector)[0]
                print(f"Catégorie prédite: {predicted_category}")
            else:
                print("Classificateur ou vectoriseur non initialisé")
                predicted_category = 'Inconnu'

            print("-" * 50)
        except Exception as e:
            print(f"Erreur pour '{query}': {str(e)}")