from extensions import db
from models import Question, Answer
from text_representation import create_tfidf_vectors
from chatbot_model import find_best_match
from app import create_app

app = create_app()

with app.app_context():
    # Extraire les données de la base de données
    questions = [q.text for q in Question.query.all()]
    answers = [a.text for a in Answer.query.all()]
    page_urls = [a.url for a in Answer.query.all()]

    # Créer les vecteurs TF-IDF
    vectorizer, tfidf_matrix = create_tfidf_vectors(questions)

    # Tester une requête
    query = "Critères d'admission à l'ISET ?"
    best_match_idx, similarity_score = find_best_match(query, questions, vectorizer, tfidf_matrix)
    print(f"Requête : {query}")
    print(f"Meilleure correspondance : {questions[best_match_idx]}")
    print(f"Réponse : {answers[best_match_idx]}")
    print(f"URL : {page_urls[best_match_idx]}")
    print(f"Score de similarité : {similarity_score:.4f}")