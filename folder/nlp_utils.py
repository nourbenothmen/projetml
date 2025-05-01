import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

nltk.download('stopwords')

class NLPProcessor:
    def __init__(self):
        self.stemmer = SnowballStemmer('french')
        self.stop_words = set(stopwords.words('french'))
        self.vectorizer = TfidfVectorizer()
        self.knn = NearestNeighbors(n_neighbors=3)
        
    def preprocess(self, text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [self.stemmer.stem(w) for w in tokens if w.isalpha() and w not in self.stop_words]
        return ' '.join(tokens)
    
    def train(self, questions):
        processed = [self.preprocess(q) for q in questions]
        self.X = self.vectorizer.fit_transform(processed)
        self.knn.fit(self.X)
    
    def find_similar(self, query):
        processed = self.preprocess(query)
        vec = self.vectorizer.transform([processed])
        distances, indices = self.knn.kneighbors(vec)
        return indices[0], distances[0]