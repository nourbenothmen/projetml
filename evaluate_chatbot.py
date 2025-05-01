# evaluate_chatbot.py
from extensions import db
from models import Log, Answer, Question
from app import create_app

app = create_app()

with app.app_context():
    logs = Log.query.all()
    total_logs = len(logs)
    
    if total_logs == 0:
        print("Aucun log disponible pour l'évaluation.")
        exit()

    # Calculate metrics
    positive_feedback = sum(1 for log in logs if log.feedback == 'positif')
    negative_feedback = sum(1 for log in logs if log.feedback == 'négatif')
    neutral_feedback = sum(1 for log in logs if log.feedback == 'neutre')
    valid_scores = [log.similarity_score for log in logs if log.similarity_score is not None]
    avg_similarity = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    # Estimate precision
    correct_matches = sum(1 for log in logs if log.similarity_score and log.similarity_score > 0.7)
    precision = correct_matches / total_logs if total_logs > 0 else 0

    # Report
    print("=== Rapport d'Évaluation du Chatbot ===")
    print(f"Total des interactions : {total_logs}")
    print(f"Feedback positif : {positive_feedback} ({positive_feedback/total_logs*100:.2f}%)")
    print(f"Feedback négatif : {negative_feedback} ({negative_feedback/total_logs*100:.2f}%)")
    print(f"Feedback neutre : {neutral_feedback} ({neutral_feedback/total_logs*100:.2f}%)")
    print(f"Score de similarité moyen : {avg_similarity:.4f}")
    print(f"Précision estimée : {precision*100:.2f}%")
    
    # Detailed analysis
    print("\nDétails des logs :")
    for log in logs:
        answer = Answer.query.get(log.answer_id)
        question = Question.query.get(answer.question_id)
        print(f"Requête utilisateur : {log.question_user}")
        print(f"Question correspondante : {question.text}")
        print(f"Réponse : {answer.text}")
        print(f"Feedback : {log.feedback or 'Aucun'}")
        print(f"Score de similarité : {log.similarity_score or 'N/A'}")
        print(f"Date : {log.date}")
        print("---")