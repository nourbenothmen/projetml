from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedChatbotModel:
    def __init__(self):
        self.word2vec = None
        self.vector_size = 100
        
    def train_word2vec(self, sentences):
        """Entraîne un modèle Word2Vec"""
        self.word2vec = Word2Vec(
            sentences=[s.split() for s in sentences],
            vector_size=self.vector_size,
            window=5,
            min_count=1,
            workers=4
        )
        self.word2vec.save("word2vec.model")
    
    def sentence_to_vector(self, sentence):
        """Convertit une phrase en vecteur moyen Word2Vec"""
        words = sentence.split()
        vectors = [self.word2vec.wv[word] for word in words if word in self.word2vec.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)
    
    def find_similar_w2v(self, query, questions, answers, threshold=0.7):
        """Recherche sémantique avec Word2Vec"""
        if not self.word2vec:
            raise ValueError("Word2Vec model not trained")
            
        query_vec = self.sentence_to_vector(preprocess_text(query))
        question_vecs = [self.sentence_to_vector(q) for q in questions]
        
        similarities = cosine_similarity([query_vec], question_vecs)
        max_index = similarities.argmax()
        max_score = similarities[0, max_index]
        
        return answers[max_index] if max_score > threshold else None