from app import app
from models import Question, Answer, Log
from extensions import db

def initialize_database():
    with app.app_context():
        db.create_all()
        
        if not Question.query.first():
            q = Question(text="Exemple question", category="test")
            a = Answer(text="Exemple réponse", question=q)
            log = Log(question_user="Test", answer_id=1, feedback="positive")
            
            db.session.add_all([q, a, log])
            db.session.commit()
            print("Base initialisée avec données de test")

if __name__ == '__main__':
    initialize_database()