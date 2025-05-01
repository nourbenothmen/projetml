from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=None,
            preprocessor=preprocess_text,
            max_features=5000
        )
    
    def train(self, questions):
        """Entraîne le vectorizer sur les questions"""
        self.vectorizer.fit(questions)
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')
    
    def vectorize(self, text):
        """Transforme un texte en vecteur TF-IDF"""
        return self.vectorizer.transform([text])
    
    def find_most_similar(self, query, questions, answers, threshold=0.6):
        """Trouve la question la plus similaire"""
        query_vec = self.vectorize(query)
        questions_vec = self.vectorizer.transform(questions)
        
        similarities = cosine_similarity(query_vec, questions_vec)
        max_index = similarities.argmax()
        max_score = similarities[0, max_index]
        
        return answers[max_index] if max_score > threshold else None