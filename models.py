from datetime import datetime
from extensions import db

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), unique=True, nullable=False, index=True)
    category = db.Column(db.String(50), nullable=False)
    answer = db.relationship('Answer', backref='question', uselist=False)

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(1000), nullable=False)
    url = db.Column(db.String(200), nullable=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)

class Log(db.Model):
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True)
    question_user = db.Column(db.String(500), nullable=False)
    answer_id = db.Column(db.Integer, db.ForeignKey('answer.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    feedback = db.Column(db.String(20), nullable=True)
    similarity_score = db.Column(db.Float, nullable=True)  # Nouveau champ
    answer = db.relationship('Answer', backref='logs')