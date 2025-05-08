from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_text

def create_tfidf_vectors(texts, language='fr'):
    preprocessed_texts = [preprocess_text(text, language=language) for text in texts]
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    return vectorizer, tfidf_matrix