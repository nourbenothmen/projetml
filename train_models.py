import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from preprocess import preprocess_text  # Importation correcte depuis preprocess.py

# Charger les données depuis chatbot_data.csv
csv_path = 'chatbot_data.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas.")

data = pd.read_csv(csv_path)
questions = data['question'].tolist()
categories = data['category'].tolist()
languages = data['language'].tolist()

# Créer des vectoriseurs TF-IDF par langue
vectorizers = {'fr': TfidfVectorizer(max_features=500, ngram_range=(1, 2)), 'en': TfidfVectorizer(max_features=500, ngram_range=(1, 2))}
tfidf_matrices = {}

for lang in ['fr', 'en']:
    lang_questions = [q for q, l in zip(questions, languages) if l == lang]
    if lang_questions:
        # Prétraiter les questions
        processed_questions = [preprocess_text(q, language=lang) for q in lang_questions]
        tfidf_matrices[lang] = vectorizers[lang].fit_transform(processed_questions)
        # Sauvegarder le vectorizer
        os.makedirs('models', exist_ok=True)
        with open(f'models/vectorizer_{lang}.pkl', 'wb') as f:
            pickle.dump(vectorizers[lang], f)
        # Sauvegarder la matrice TF-IDF (facultatif, mais utile pour accélérer)
        with open(f'models/tfidf_matrix_{lang}.pkl', 'wb') as f:
            pickle.dump(tfidf_matrices[lang], f)
        print(f"Vectoriser et matrice TF-IDF sauvegardés pour {lang}")

# Entraîner le classificateur de catégories
# Utiliser les questions prétraitées de la langue principale (fr) pour simplifier
category_questions = [preprocess_text(q, language=l) for q, l in zip(questions, languages) if l in ['fr', 'en']]
if category_questions and categories:
    X_category = vectorizers['fr'].transform(category_questions)  # Utiliser le vectorizer fr
    category_classifier = MultinomialNB()
    category_classifier.fit(X_category, categories)
    # Sauvegarder le classificateur
    os.makedirs('models/category_classifier', exist_ok=True)
    with open('models/category_classifier/model.pkl', 'wb') as f:
        pickle.dump(category_classifier, f)
    # Sauvegarder le vectorizer associé
    with open('models/category_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizers['fr'], f)
    print("Classificateur de catégories entraîné et sauvegardé")
else:
    print("Aucune donnée de catégorie disponible pour l'entraînement.")

# Optionnel : Entraîner un classificateur d'intention si des données sont disponibles
intents = data['intent'].tolist()
if any(i for i in intents if i):  # Vérifier s'il y a des intentions non vides
    intent_questions = [preprocess_text(q, language=l) for q, l, i in zip(questions, languages, intents) if i]
    X_intent = vectorizers['fr'].transform(intent_questions)
    intent_classifier = MultinomialNB()
    intent_classifier.fit(X_intent, [i for i in intents if i])
    with open('models/intent_classifier/model.pkl', 'wb') as f:
        pickle.dump(intent_classifier, f)
    with open('models/intent_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizers['fr'], f)
    print("Classificateur d'intention entraîné et sauvegardé")
else:
    print("Aucune donnée d'intention disponible pour l'entraînement.")

print("Entraînement terminé.")