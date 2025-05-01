from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_migrate import Migrate
from extensions import db
from models import Question, Answer, Log
from text_representation import create_tfidf_vectors
from chatbot_model import find_best_match

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://chatbot_nour:nour782@localhost/chatbot-iset'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    db.init_app(app)
    Migrate(app, db)
    
    # Create tables if they don't exist
    with app.app_context():
        db.create_all()  # Ensure tables are created
        # Load TF-IDF vectors only if questions exist
        questions = [q.text for q in Question.query.all()]
        app.vectorizer, app.tfidf_matrix = create_tfidf_vectors(questions) if questions else (None, None)
        app.questions = questions
        app.answers = [(a.text, a.url, a.id) for a in Answer.query.all()]
    
    @app.route('/api/questions', methods=['POST'])
    def create_question(): 
        data = request.get_json()
        required_fields = ['question', 'answer', 'category']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Données manquantes. Requis: question, answer, category'}), 400
        
        try:
            new_question = Question(
                text=data['question'],
                category=data['category']
            )
            db.session.add(new_question)
            db.session.flush()
        
            new_answer = Answer(
                text=data['answer'],
                url=data.get('url', ''),
                question_id=new_question.id
            )
            db.session.add(new_answer)
            db.session.commit()
            
            app.questions.append(new_question.text)
            app.answers.append((new_answer.text, new_answer.url, new_answer.id))
            app.vectorizer, app.tfidf_matrix = create_tfidf_vectors(app.questions)
            
            return jsonify({
                'status': 'success',
                'message': 'Question et réponse ajoutées',
                'question_id': new_question.id
            }), 201
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/questions', methods=['GET'])
    def get_questions():
        questions = Question.query.all()
        return jsonify([{
            'id': q.id,
            'text': q.text,
            'category': q.category
        } for q in questions])
    
    @app.route('/api/ask', methods=['POST'])
    def ask_question():
        data = request.get_json()
        question_text = data.get('question')
        
        if not question_text:
            return jsonify({'error': 'Aucune question fournie'}), 400
        
        if not app.questions or not app.vectorizer:
            return jsonify({'error': 'Aucune question disponible dans la base de données'}), 500
        
        best_match_idx, similarity_score = find_best_match(
            question_text, app.questions, app.vectorizer, app.tfidf_matrix
        )
        answer_text, answer_url, answer_id = app.answers[best_match_idx]
        
        new_log = Log(
            question_user=question_text,
            answer_id=answer_id,
            feedback=None,
            similarity_score=similarity_score
        )
        db.session.add(new_log)
        db.session.commit()
        
        question = Question.query.filter_by(text=app.questions[best_match_idx]).first()
        
        return jsonify({
            'answer': answer_text,
            'url': answer_url,
            'category': question.category,
            'similarity_score': float(similarity_score),
            'answer_id': answer_id
        })
    
    @app.route('/api/log-feedback', methods=['POST'])
    def log_feedback():
        data = request.get_json()
        question = data.get('question')
        answer_id = data.get('answer_id')
        feedback = data.get('feedback')
        
        if not all([question, answer_id, feedback]):
            return jsonify({'error': 'Données manquantes'}), 400
        
        new_log = Log(
            question_user=question,
            answer_id=answer_id,
            feedback=feedback,
            similarity_score=None
        )
        db.session.add(new_log)
        db.session.commit()
        
        return jsonify({"status": "success", "message": "Feedback enregistré"})
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)