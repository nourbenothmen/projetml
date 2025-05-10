from flask import Flask, request, jsonify ,send_from_directory
from flask_cors import CORS
from extensions import db
from models import User
from text_representation import create_tfidf_vectors, preprocess_text
from chatbot_model import find_best_match
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import csv
import jwt
import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import re
import os
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup

# --- Nouvelle fonction pour charger les données de iset_sfax_qna.csv ---
def load_iset_sfax_qna(app_instance):
    """
    Charge les questions et réponses de iset_sfax_qna.csv dans un dictionnaire.
    Structure : {langue: {question: {'answer': ..., 'category': ..., 'url': ..., 'intent': ...}}}
    """
    qna_path = 'iset_sfax_qna.csv'  # Chemin du fichier iset_sfax_qna.csv
    app_instance.iset_sfax_qna = {'fr': {}, 'en': {}}  # Initialisation du dictionnaire

    if not os.path.exists(qna_path):
        print(f"Le fichier {qna_path} n'existe pas. Aucune donnée supplémentaire ne sera chargée.")
        return

    try:
        with open(qna_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            expected_fieldnames = ['question', 'answer', 'category', 'language', 'url', 'intent']
            if not reader.fieldnames or set(reader.fieldnames) != set(expected_fieldnames):
                raise ValueError(f"Le fichier CSV {qna_path} doit avoir les en-têtes suivants : {expected_fieldnames}")

            for row in reader:
                try:
                    question = row['question'].strip()
                    lang = row['language'].strip()
                    answer = row['answer'].strip()
                    category = row['category'].strip()
                    url = row.get('url', '').strip()
                    intent = row.get('intent', '').strip()

                    if not question or not answer or not category or not lang:
                        print(f"Ignoré: Ligne incomplète dans iset_sfax_qna.csv - {row}")
                        continue

                    if lang not in ['fr', 'en']:
                        print(f"Ignoré: Langue invalide '{lang}' pour question dans iset_sfax_qna.csv: {question}")
                        continue

                    app_instance.iset_sfax_qna[lang][question] = {
                        'answer': answer,
                        'category': category,
                        'url': url,
                        'intent': intent
                    }
                    print(f"Chargé dans iset_sfax_qna[{lang}]: {question}")
                except KeyError as e:
                    print(f"Erreur: Clé manquante {e} dans la ligne de iset_sfax_qna.csv: {row}")
                    continue
                except Exception as e:
                    print(f"Erreur inattendue dans la ligne de iset_sfax_qna.csv: {row}, erreur: {str(e)}")
                    continue

        print(f"Total questions chargées depuis iset_sfax_qna.csv: fr={len(app_instance.iset_sfax_qna['fr'])}, en={len(app_instance.iset_sfax_qna['en'])}")
    except Exception as e:
        print(f"Erreur lors du chargement de iset_sfax_qna.csv: {str(e)}")

# --- Fonction pour trouver une correspondance dans iset_sfax_qna.csv ---
def find_in_iset_sfax_qna(query: str, language: str, app_instance) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], float]:
    """
    Recherche la question la plus similaire dans iset_sfax_qna.csv.
    Retourne (answer, category, url, intent, similarity_score).
    """
    if language not in app_instance.iset_sfax_qna or not app_instance.iset_sfax_qna[language]:
        print(f"Aucune donnée disponible dans iset_sfax_qna pour la langue {language}")
        return None, None, None, None, 0.0

    questions = list(app_instance.iset_sfax_qna[language].keys())
    if not questions:
        print(f"Aucune question trouvée dans iset_sfax_qna pour la langue {language}")
        return None, None, None, None, 0.0

    # Créer un vectorizer temporaire pour iset_sfax_qna si nécessaire
    if not hasattr(app_instance, 'iset_sfax_vectorizers'):
        app_instance.iset_sfax_vectorizers = {}
        app_instance.iset_sfax_tfidf_matrices = {}

    if language not in app_instance.iset_sfax_vectorizers:
        app_instance.iset_sfax_vectorizers[language], app_instance.iset_sfax_tfidf_matrices[language] = create_tfidf_vectors(
            questions, language=language
        )
        print(f"TF-IDF créé pour iset_sfax_qna[{language}]: {app_instance.iset_sfax_tfidf_matrices[language].shape}")

    # Calculer la similarité
    processed_query = preprocess_text(query, language=language)
    query_vector = app_instance.iset_sfax_vectorizers[language].transform([processed_query])
    similarities = cosine_similarity(query_vector, app_instance.iset_sfax_tfidf_matrices[language])[0]
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    best_question = questions[best_idx]

    print(f"Meilleure correspondance dans iset_sfax_qna: question='{best_question}', score={best_score}")

    if best_score >= 0.8:  # Seuil pour considérer une correspondance valide
        data = app_instance.iset_sfax_qna[language][best_question]
        return data['answer'], data['category'], data['url'], data['intent'], best_score
    else:
        print(f"Score de similarité trop bas ({best_score}) dans iset_sfax_qna")
        return None, None, None, None, 0.0

# --- Fonctions existantes (non modifiées) ---
def load_data_from_csv(app_instance):
    expected_fieldnames = ['question', 'answer', 'category', 'language', 'url', 'intent', 'answer_type']
    csv_path = app_instance.config['CSV_PATH']
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas.")
    
    app_instance.questions = {'fr': [], 'en': []}
    app_instance.answers = {'fr': [], 'en': []}
    seen_questions = set()
    
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or set(reader.fieldnames) != set(expected_fieldnames):
                raise ValueError(f"Le fichier CSV {csv_path} doit avoir les en-têtes suivants : {expected_fieldnames}")
            
            for row in reader:
                try:
                    question = row['question'].strip()
                    lang = row['language'].strip()
                    answer = row['answer'].strip()
                    category = row['category'].strip()
                    url = row.get('url', '').strip()
                    intent = row.get('intent', '').strip()
                    answer_type = row.get('answer_type', 'text').strip()  # Par défaut, "text" si non spécifié
                    
                    if not question or not answer or not category or not lang:
                        print(f"Ignoré: Ligne incomplète - {row}")
                        continue
                    
                    if lang not in ['fr', 'en']:
                        print(f"Ignoré: Langue invalide '{lang}' pour question: {question}")
                        continue
                    
                    if answer_type not in ['text', 'image']:
                        print(f"Ignoré: Type de réponse invalide '{answer_type}' pour question: {question}")
                        continue
                    
                    if question not in seen_questions:
                        seen_questions.add(question)
                        app_instance.questions[lang].append(question)
                        # Stocker answer_type avec la réponse
                        app_instance.answers[lang].append((answer, url, len(app_instance.answers[lang]), answer_type))
                        print(f"Chargé: {question} (lang: {lang}, category: {category}, url: {url}, answer_type: {answer_type})")
                    else:
                        print(f"Ignoré: Question en double: {question}")
                except KeyError as e:
                    print(f"Erreur: Clé manquante {e} dans la ligne: {row}")
                    continue
                except Exception as e:
                    print(f"Erreur inattendue dans la ligne: {row}, erreur: {str(e)}")
                    continue
    
        print(f"Total questions chargées: fr={len(app_instance.questions['fr'])}, en={len(app_instance.questions['en'])}")
    
    except UnicodeDecodeError:
        print(f"Erreur: Problème d'encodage dans {csv_path}. Assurez-vous qu'il est en UTF-8 sans BOM.")
        raise
    except Exception as e:
        print(f"Erreur lors du chargement de {csv_path}: {str(e)}")
        raise
    
    for lang in ['fr', 'en']:
        if app_instance.questions[lang]:
            try:
                app_instance.vectorizers[lang], app_instance.tfidf_matrices[lang] = create_tfidf_vectors(
                    app_instance.questions[lang], language=lang
                )
                print(f"TF-IDF pour {lang}: {app_instance.tfidf_matrices[lang].shape}")
            except Exception as e:
                print(f"Erreur lors de la création des vecteurs TF-IDF pour {lang}: {str(e)}")
                raise
        else:
            print(f"Aucune question pour la langue {lang}")
def train_category_classifier(app):
    questions = []
    categories = []
    try:
        with open(app.config['CSV_PATH'], newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    if row['category']:
                        lang = row['language']
                        processed_question = preprocess_text(row['question'], language=lang)
                        questions.append(processed_question)
                        categories.append(row['category'])
                        print(f"Ajouté pour entraînement: {row['question']} -> {row['category']} (prétraité: {processed_question})")
                except KeyError as e:
                    print(f"Erreur: Clé manquante {e} dans la ligne: {row}")
                    continue
                except Exception as e:
                    print(f"Erreur inattendue dans la ligne: {row}, erreur: {str(e)}")
                    continue
    
        if questions and categories:
            app.category_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X = app.category_vectorizer.fit_transform(questions)
            app.category_classifier = MultinomialNB()
            app.category_classifier.fit(X, categories)
            print(f"Classificateur de catégories entraîné avec succès. Catégories: {set(categories)}")
        else:
            print("Aucune donnée de catégorie disponible pour l'entraînement.")
            app.category_classifier = None
            app.category_vectorizer = None
    except Exception as e:
        print(f"Erreur lors de l'entraînement du classificateur: {str(e)}")
        raise

def predict_category(query, vectorizer, classifier, language='fr'):
    try:
        preprocessed_query = preprocess_text(query, language=language)
        query_vector = vectorizer.transform([preprocessed_query])
        predicted = classifier.predict(query_vector)[0]
        print(f"Catégorie prédite pour '{query}': {predicted}")
        return predicted
    except Exception as e:
        print(f"Erreur lors de la prédiction de catégorie pour '{query}': {str(e)}")
        return 'Inconnu'

def save_to_csv(question: str, answer: str, category: str, language: str, url: str = '', intent: str = '', answer_type: str = 'text'):
    try:
        # Validate inputs
        if not all([question, answer, category, language]):
            raise ValueError(f"Champ manquant: question={question}, answer={answer}, category={category}, language={language}")
        if language not in ['fr', 'en']:
            raise ValueError(f"Langue invalide: {language}")
        if answer_type not in ['text', 'image']:
            raise ValueError(f"Type de réponse invalide: {answer_type}")
        
        # Vérifier si la question existe déjà pour éviter les doublons
        if question in app.questions[language]:
            print(f"Question déjà existante dans app.questions[{language}]: {question}, mise à jour ignorée")
            return
        
        # Read existing rows
        rows = []
        with open(app.config['CSV_PATH'], newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            if not fieldnames or set(fieldnames) != set(['question', 'answer', 'category', 'language', 'url', 'intent', 'answer_type']):
                raise ValueError(f"En-têtes CSV invalides: {fieldnames}")
            for row in reader:
                sanitized_row = {k: row.get(k, '') for k in fieldnames}
                rows.append(sanitized_row)
        
        # Append new row
        new_row = {
            'question': question,
            'answer': answer,
            'category': category,
            'language': language,
            'url': url,
            'intent': intent,
            'answer_type': answer_type  # Ajout du champ answer_type
        }
        print(f"Écriture dans CSV: {new_row}")
        rows.append(new_row)
        
        # Write back to CSV
        with open(app.config['CSV_PATH'], 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['question', 'answer', 'category', 'language', 'url', 'intent', 'answer_type'])
            writer.writeheader()
            writer.writerows(rows)
        
        # Update app state
        app.questions[language].append(question)
        app.answers[language].append((answer, url, len(app.answers[language]), answer_type))
        app.vectorizers[language], app.tfidf_matrices[language] = create_tfidf_vectors(app.questions[language], language=language)
        print(f"Sauvegardé: {question} (lang: {language}, category: {category}, answer_type: {answer_type})")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde dans le CSV: {str(e)}")
        raise
def append_feedback(question: str, answer_id: int, feedback: str, similarity_score: Optional[float] = None):
    try:
        fieldnames = ['question', 'answer_id', 'feedback', 'timestamp', 'similarity_score']
        file_exists = os.path.isfile(app.config['FEEDBACK_PATH'])
        with open(app.config['FEEDBACK_PATH'], 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'question': question,
                'answer_id': answer_id,
                'feedback': feedback,
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'similarity_score': similarity_score if similarity_score is not None else ''
            })
        print(f"Feedback ajouté: {question}, feedback: {feedback}")
    except Exception as e:
        print(f"Erreur lors de l'ajout du feedback: {str(e)}")
        raise

def adjust_similarity_threshold() -> float:
    try:
        if not os.path.exists(app.config['FEEDBACK_PATH']):
            print("Aucun fichier de feedback trouvé, seuil par défaut maintenu.")
            return app.similarity_threshold
        with open(app.config['FEEDBACK_PATH'], newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            positive_feedbacks = [float(row['similarity_score']) for row in reader if row['feedback'] == 'positive' and row['similarity_score']]
            if positive_feedbacks:
                new_threshold = max(0.3, sum(positive_feedbacks) / len(positive_feedbacks) - 0.1)
                print(f"Seuil de similarité ajusté à: {new_threshold}")
                return new_threshold
            print("Aucun feedback positif avec score valide trouvé.")
        return app.similarity_threshold
    except Exception as e:
        print(f"Erreur lors de l'ajustement du seuil: {str(e)}")
        return app.similarity_threshold

def retrain_intent_classifier():
    try:
        questions_for_intent = []
        intents = []
        with open(app.config['CSV_PATH'], newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('intent'):
                    questions_for_intent.append(row['question'])
                    intents.append(row['intent'])
        if questions_for_intent and intents:
            app.intent_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X_intent = app.intent_vectorizer.fit_transform(questions_for_intent)
            app.intent_classifier = MultinomialNB()
            app.intent_classifier.fit(X_intent, intents)
            print("Modèle d'intention réentraîné avec succès.")
        else:
            print("Aucune donnée valide pour réentraîner le modèle.")
    except Exception as e:
        print(f"Erreur lors du réentraînement du modèle d'intention: {str(e)}")
        raise

def map_query_to_url(query: str) -> str:
    query_lower = query.lower()
    url_mappings = {
        'inscription': 'https://isetsf.rnu.tn/fr/inscription',
        'formation': 'https://isetsf.rnu.tn/fr/formation',
        'cours': 'https://isetsf.rnu.tn/fr/formation',
        'licence': 'https://isetsf.rnu.tn/fr/formation/licences',
        'mastère': 'https://isetsf.rnu.tn/fr/candidature-premiere-annee-mastere',
        'club': 'https://isetsf.rnu.tn/fr/clubs',
        'stage': 'https://isetsf.rnu.tn/fr/offres-de-stage',
        'bourse': 'https://isetsf.rnu.tn/fr/bourses',
        'international': 'https://isetsf.rnu.tn/fr/international',
        'recherche': 'https://isetsf.rnu.tn/fr/recherche',
        'bibliothèque': 'https://isetsf.rnu.tn/fr/bibliotheque',
        'admission': 'https://isetsf.rnu.tn/fr/admissions',
        'contact': 'https://isetsf.rnu.tn/fr/contact',
        'adresse': 'https://isetsf.rnu.tn/fr/contact',
        'alumni': 'https://isetsf.rnu.tn/fr/alumni',
        'manifestation': 'https://isetsf.rnu.tn/fr/manifestations',
        'vie estudiantine': 'https://isetsf.rnu.tn/fr/vie-estudiantine',
        'partenaires professionnels': 'https://isetsf.rnu.tn/fr/actualites/partenaires-professionnels',
        'informatique': 'https://isetsf.rnu.tn/fr/institut/departements/technologies-de-l-informatique',
        'génie civil': 'https://isetsf.rnu.tn/fr/institut/departements/genie-civil',
        'comité pédagogique': 'https://isetsf.rnu.tn/fr/institut/departements/technologies-de-l-informatique',
        'présentation département': 'https://isetsf.rnu.tn/fr/institut/departements/technologies-de-l-informatique'
    }
    if 'directeur' in query_lower and 'informatique' in query_lower:
        print("URL mappé pour 'directeur informatique': https://isetsf.rnu.tn/fr/institut/departements/technologies-de-l-informatique")
        return 'https://isetsf.rnu.tn/fr/institut/departements/technologies-de-l-informatique'
    for keyword, url in url_mappings.items():
        if keyword in query_lower:
            print(f"URL mappé pour '{keyword}': {url}")
            return url
    print("Aucun mot-clé trouvé, utilisation de l'URL par défaut")
    return 'https://isetsf.rnu.tn/fr'

def scrape_iset_website(url: str, query: str, language: str = 'fr') -> Tuple[Optional[str], str]:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        print(f"Tentative de scraping de l'URL: {url}")
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, 'html.parser')
        
        texts = []
        committee_texts = set()
        committee_keywords = ['comité pédagogique', 'responsable', 'coordinateur']
        exclude_keywords = ['secrétaire', 'directeur']
        
        for elem in soup.find_all(['p', 'div', 'ul', 'li', 'table', 'tr', 'td']):
            text = ' '.join(elem.stripped_strings)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 10 and any(keyword in text.lower() for keyword in committee_keywords) and not any(exclude_keyword in text.lower() for exclude_keyword in exclude_keywords):
                committee_texts.add(text)
                texts.append(text)
                print(f"Texte trouvé pour comité pédagogique: {text[:100]}...")
            elif len(text) > 10:
                texts.append(text)
        
        if not texts:
            print(f"Aucun texte pertinent trouvé sur {url}")
            return ("Les membres du comité pédagogique du département Technologies de l'Informatique ne sont pas indiqués sur la page. Veuillez contacter le département au +216 74 431 425 pour plus d'informations.", url)
        
        print(f"Textes extraits avant filtrage: {[t[:100] + '...' for t in texts]}")
        
        if not app.vectorizers.get(language):
            print(f"Erreur: Vectorizer non initialisé pour la langue {language}")
            return ("Erreur interne: Vectorizer non initialisé.", url)
        
        processed_query = preprocess_text(query, language=language)
        query_vector = app.vectorizers[language].transform([processed_query])
        text_vectors = app.vectorizers[language].transform([preprocess_text(t, language) for t in texts])
        similarities = cosine_similarity(query_vector, text_vectors)[0]
        best_idx = similarities.argmax()
        
        answer = None
        if 'comité pédagogique' in query.lower():
            all_names = set()  # Utiliser un set pour éviter les doublons
            for text in committee_texts:
                print(f"Analyse du texte pour extraction des noms: {text}")
                # Extraire les noms complets, ignorer les rôles
                name_matches = re.findall(r'(?:Mme\.?|M\.)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+)(?:\s+(?:Responsable|Coordinateur|MASTER\s+[A-Z]+)?)?', text)
                for name, *_ in name_matches:
                    print(f"Nom extrait: {name}")
                    if 'Amel BOURICHA' not in name and 'Mohamed ELLEUCH' not in name:
                        all_names.add(name.strip())
            
            if all_names:
                formatted_names = []
                for name in sorted(all_names):
                    title = 'Mme' if 'Afef' in name else 'M.'
                    formatted_names.append(f"{title} {name}")
                answer = "Membres du comité pédagogique : " + ", ".join(formatted_names)
                print(f"Réponse construite à partir des noms extraits: {answer}")
            else:
                print(f"Aucun nom propre détecté dans les textes du comité")
                answer = ("Les membres du comité pédagogique du département Technologies de l'Informatique ne sont pas indiqués sur la page. Veuillez contacter le département au +216 74 431 425 pour plus d'informations.")
        
        else:
            answer = texts[best_idx][:500] + '...' if len(texts[best_idx]) > 500 else texts[best_idx]
        
        print(f"Réponse extraite de {url}: {answer}")
        return answer, url
    except Exception as e:
        print(f"Erreur lors du scraping de {url}: {str(e)}")
        return None, url

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception as e:
        print(f"Erreur lors de la détection de la langue: {str(e)}")
        return 'fr'  # Langue par défaut

# Définitions minimales pour les fonctions manquantes
def predict_intent(query: str, language: str) -> Optional[str]:
    print(f"predict_intent non implémenté, retour de None pour query: {query}")
    return None

def extract_entities(query: str, language: str) -> dict:
    entities = {'course': 'informatique'} if 'informatique' in query.lower() else {}
    print(f"Entités extraites: {entities}")
    return entities

def load_questions(language: str) -> dict:
    questions_dict = {}
    if language in app.questions:
        for idx, question in enumerate(app.questions[language]):
            answer, url, answer_id, answer_type = app.answers[language][idx]  # Ajout de answer_type
            questions_dict[question] = {'answer': answer, 'url': url, 'id': answer_id, 'answer_type': answer_type}
    print(f"Questions chargées pour {language}: {list(questions_dict.keys())}")
    return questions_dict
def find_most_similar_question(query: str, questions: dict, language: str) -> Tuple[int, float]:
    if not questions or app.vectorizers.get(language) is None or app.tfidf_matrices.get(language) is None:
        print("Aucune question ou vectorizer/matrice non initialisé, retour de (-1, 0.0)")
        return -1, 0.0
    question_texts = list(questions.keys())
    query_vector = app.vectorizers[language].transform([query])
    similarities = cosine_similarity(query_vector, app.tfidf_matrices[language])[0]
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    print(f"Meilleure correspondance: index={best_idx}, score={best_score}")
    return best_idx, best_score

def create_app():
    app = Flask(__name__, template_folder='templates')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://chatbot-nour:nour782@localhost:5432/chatbot-iset'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'static-secret-key-for-testing'
    app.config['CSV_PATH'] = 'chatbot_data.csv'
    app.config['FEEDBACK_PATH'] = 'feedbacks.csv'

    CORS(app, resources={r"/api/*": {"origins": "*"}})
    db.init_app(app)

    app.questions: Dict[str, List[str]] = {'fr': [], 'en': []}
    app.answers: Dict[str, List[Tuple[str, str, int]]] = {'fr': [], 'en': []}
    app.vectorizers: Dict[str, Optional[TfidfVectorizer]] = {'fr': None, 'en': None}
    app.tfidf_matrices: Dict[str, Optional[any]] = {'fr': None, 'en': None}
    app.intent_vectorizer: Optional[TfidfVectorizer] = None
    app.intent_classifier: Optional[any] = None
    app.category_vectorizer: Optional[TfidfVectorizer] = None
    app.category_classifier: Optional[MultinomialNB] = None
    app.similarity_threshold: float = 0.5
    app.iset_sfax_qna: Dict[str, Dict[str, Dict]] = {'fr': {}, 'en': {}}  # Nouvelle structure pour iset_sfax_qna

    try:
        load_data_from_csv(app)
        train_category_classifier(app)
        load_iset_sfax_qna(app)  # Charger les données de iset_sfax_qna.csv
    except Exception as e:
        print(f"Erreur lors de l'initialisation de l'application: {str(e)}")
        raise

    print(f"Questions chargées: {app.questions}")
    print(f"Réponses chargées: {app.answers}")
    print(f"Données iset_sfax_qna chargées: {app.iset_sfax_qna}")

    def token_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = None
            auth_header = request.headers.get('Authorization')
            print(f"Auth header reçu: {auth_header}")
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(" ")[1]
                print(f"Token extrait: {token}")
            else:
                print("Erreur: En-tête Authorization manquant ou mal formé")
                return jsonify({'error': 'Token manquant ou mal formé'}), 401
            try:
                data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
                current_user = str(data['sub'])
                print(f"Utilisateur authentifié: {current_user}")
            except jwt.ExpiredSignatureError as e:
                print(f"Erreur JWT: Token expiré - {str(e)}")
                return jsonify({'error': 'Token expiré'}), 401
            except jwt.InvalidTokenError as e:
                print(f"Erreur JWT: Token invalide - {str(e)}")
                return jsonify({'error': f'Token invalide: {str(e)}'}), 401
            except Exception as e:
                print(f"Erreur inattendue dans JWT: {str(e)}")
                return jsonify({'error': f'Erreur JWT inattendue: {str(e)}'}), 401
            return f(current_user, *args, **kwargs)
        return decorated

    @app.route('/api/register', methods=['POST'])
    def register():
        try:
            data = request.get_json()
            print(f"Données reçues pour inscription: {data}")
            if not data or not all(k in data for k in ['username', 'password']):
                return jsonify({'error': 'Nom d’utilisateur et mot de passe requis'}), 400
            username = data['username']
            password = data['password']
            if not username or not password:
                return jsonify({'error': 'Nom d’utilisateur et mot de passe ne peuvent pas être vides'}), 400
            if User.query.filter_by(username=username).first():
                return jsonify({'error': 'Nom d’utilisateur déjà pris'}), 400
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            print(f"Utilisateur {username} créé avec succès")
            return jsonify({'message': 'Utilisateur créé avec succès'}), 201
        except Exception as e:
            db.session.rollback()
            print(f"Erreur lors de l’inscription: {str(e)}")
            return jsonify({'error': f'Erreur lors de l’inscription: {str(e)}'}), 400

    @app.route('/api/login', methods=['POST'])
    def login():
        try:
            data = request.get_json()
            print(f"Données reçues pour connexion: {data}")
            if not data or not all(k in data for k in ['username', 'password']):
                return jsonify({'error': 'Nom d’utilisateur et mot de passe requis'}), 400
            username = data['username']
            password = data['password']
            user = User.query.filter_by(username=username).first()
            if not user or not user.check_password(password):
                return jsonify({'error': 'Nom d’utilisateur ou mot de passe incorrect'}), 401
            token = jwt.encode({
                'sub': str(user.id),
                'iat': datetime.datetime.utcnow(),
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm='HS256')
            print(f"Token généré pour {username}: {token}")
            return jsonify({'token': token}), 200
        except Exception as e:
            print(f"Erreur lors de la connexion: {str(e)}")
            return jsonify({'error': f'Erreur lors de la connexion: {str(e)}'}), 400

    @app.route('/api/ask', methods=['POST'])
    def ask():
        try:
            request_id = datetime.datetime.utcnow().isoformat()
            print(f"Nouvelle requête reçue à /api/ask (ID: {request_id}): question='{request.get_json().get('question', '')}'")
    
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
    
            question = data.get('question', '').strip()
            language = data.get('language', 'fr').strip()
    
            if not question:
                return jsonify({'error': 'Question is required'}), 400
    
            print(f"Question reçue: {question}, Langue préférée: {language}")
    
            detected_language = detect_language(question)
            final_language = language if language in ['fr', 'en', 'ar'] else detected_language
            print(f"Langue détectée: {detected_language}, Langue finale: {final_language}")
    
            processed_query = preprocess_text(question, language=final_language)
            print(f"Query prétraitée: {processed_query}")
    
            category = predict_category(processed_query, app.category_vectorizer, app.category_classifier, final_language)
            print(f"Catégorie prédite pour '{processed_query}': {category}")
    
            intent = predict_intent(processed_query, final_language)
            print(f"Intention prédite: {intent}")
    
            entities = extract_entities(processed_query, final_language)
            print(f"Entités détectées: {entities}")
    
            available_questions = load_questions(final_language)
            print(f"Questions disponibles pour {final_language}: {list(available_questions.keys())}")
    
            best_index, similarity_score = find_most_similar_question(processed_query, available_questions, final_language)
            print(f"Meilleur index: {best_index}, Score de similarité: {similarity_score}")
    
            similarity_threshold = 0.9
            if 'comité pédagogique' in question.lower():
                similarity_threshold = 0.95
            print(f"Seuil de similarité: {similarity_threshold}")
    
            answer_id = None
            answer = None
            url = None
            answer_type = 'text'  # Par défaut
    
            # Traitement de la réponse
            if best_index != -1 and similarity_score >= similarity_threshold:
                similar_question = list(available_questions.keys())[best_index]
                print(f"Question correspondante: {similar_question}")
                answer_data = available_questions[similar_question]
                answer = answer_data['answer']
                url = answer_data.get('url', '')
                answer_id = answer_data.get('id', None)
                answer_type = app.answers[final_language][best_index][3]  # Récupérer answer_type
                print(f"Réponse trouvée dans chatbot_data.csv: {answer}, URL: {url}, ID: {answer_id}, Type: {answer_type}")
            else:
                print(f"Aucune correspondance trouvée dans chatbot_data.csv, recherche dans iset_sfax_qna.csv...")
                qna_answer, qna_category, qna_url, qna_intent, qna_similarity = find_in_iset_sfax_qna(question, final_language, app)
            
                if qna_answer:
                    print(f"Réponse trouvée dans iset_sfax_qna.csv: {qna_answer}")
                    answer = qna_answer
                    category = qna_category if qna_category else category
                    url = qna_url if qna_url else ''
                    intent = qna_intent if qna_intent else intent
                    answer_id = len(app.questions[final_language])
                    similarity_score = qna_similarity
                    answer_type = 'text'  # Réponses de iset_sfax_qna sont toujours du texte dans cet exemple

                    print(f"Ajout de la question-réponse à chatbot_data.csv: {question}")
                    save_to_csv(question, answer, category, final_language, url, intent or '', answer_type)
                else:
                    print(f"Aucune correspondance trouvée dans iset_sfax_qna.csv, tentative de scraping...")
                    mapped_url = map_query_to_url(question)
                    print(f"URL mappée: {mapped_url}")
                    answer, url = scrape_iset_website(mapped_url, question, final_language)
                    if not answer:
                        answer = "Désolé, je n'ai pas trouvé de réponse pertinente. Veuillez reformuler votre question ou consulter le site officiel de l'ISET Sfax : https://isetsf.rnu.tn/fr"
                    url = mapped_url
                    answer_id = len(app.questions[final_language])
                    print(f"Préparation de l'enregistrement dans chatbot_data.csv pour question: {question}")
                    save_to_csv(question, answer, category, final_language, url, intent or '', 'text')
                    print(f"Sauvegardé: {question} (lang: {final_language}, category: {category})")
    
            # Ajout du feedback
            print(f"Préparation de l'ajout du feedback pour question: {question}")
            append_feedback(question, answer_id or 0, 'pending', similarity_score)
            print(f"Feedback ajouté: {question}, feedback: pending")
    
            # Construction de la réponse
            response = {
                'answer': answer,
                'url': url,
                'category': category,
                'similarity_score': similarity_score,
                'answer_id': answer_id,
                'language': final_language,
                'intent': intent,
                'entities': entities,
                'answer_type': answer_type  # Ajout du type de réponse
            }
    
            print(f"Réponse finale envoyée (ID: {request_id}): {response}")
            return jsonify(response)

        except Exception as e:
            print(f"Erreur dans /api/ask (ID: {request_id}): {str(e)}")
            return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500
    @app.route('/images/<path:filename>')
    def serve_image(filename):
        try:
            return send_from_directory('images', filename)
        except FileNotFoundError:
            print(f"Fichier non trouvé: {filename}")
            return "Image non trouvée", 404
    
    @app.route('/api/suggest', methods=['POST'])
    @token_required
    def suggest_question(current_user):
        try:
            data = request.get_json()
            if not data or not all(k in data for k in ['question', 'answer', 'language', 'category']):
                return jsonify({'error': 'Question, réponse, langue et catégorie requis'}), 400
            lang = data['language']
            if lang not in ['fr', 'en']:
                return jsonify({'error': 'Langue doit être "fr" ou "en"'}), 400
            question = data['question']
            answer = data['answer']
            category = data['category']
            url = data.get('url', '')
            intent = data.get('intent', '')

            app.questions[lang].append(question)
            app.answers[lang].append((answer, url, len(app.answers[lang])))
            app.vectorizers[lang], app.tfidf_matrices[lang] = create_tfidf_vectors(app.questions[lang], language=lang)
            save_to_csv(question, answer, category, lang, url, intent)
            return jsonify({
                'status': 'success',
                'message': 'Suggestion ajoutée et persistée',
                'question_id': len(app.questions[lang]) - 1
            }), 201
        except Exception as e:
            print(f"Erreur dans /api/suggest: {str(e)}")
            return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

    @app.route('/api/feedback', methods=['POST'])
    @token_required
    def log_feedback(current_user):
        try:
            data = request.get_json()
            if not data or not all(k in data for k in ['question', 'answer_id', 'feedback']):
                return jsonify({'error': 'Question, answer_id et feedback requis'}), 400
            append_feedback(data['question'], data['answer_id'], data['feedback'], data.get('similarity_score'))
            return jsonify({'status': 'success', 'message': 'Feedback enregistré'})
        except Exception as e:
            print(f"Erreur dans /api/feedback: {str(e)}")
            return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

    @app.route('/api/validate/<int:question_id>', methods=['POST'])
    @token_required
    def validate_suggestion(current_user):
        try:
            data = request.get_json()
            if not data or 'action' not in data:
                return jsonify({'error': 'Action requise (approve/reject)'}), 400
            action = data['action']
            lang = next((l for l in ['fr', 'en'] if question_id < len(app.questions[l])), 'fr')
            if question_id >= len(app.questions[lang]):
                return jsonify({'error': 'ID de question invalide'}), 404
            question = app.questions[lang][question_id]
            answer, url, _ = app.answers[lang][question_id]
            category = 'validated' if action == 'approve' else 'rejected'
            if action == 'approve':
                save_to_csv(question, answer, category, lang, url, next((row['intent'] for row in csv.DictReader(open(app.config['CSV_PATH'])) if row['question'] == question and row['language'] == lang), ''))
                retrain_intent_classifier()
            elif action == 'reject':
                del app.questions[lang][question_id]
                del app.answers[lang][question_id]
                with open(app.config['CSV_PATH'], 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['question', 'answer', 'category', 'language', 'url', 'intent'])
                    writer.writeheader()
                    for q, (a, u, _) in zip(app.questions[lang], app.answers[lang]):
                        writer.writerow({'question': q, 'answer': a, 'category': 'validated', 'language': lang, 'url': u, 'intent': ''})
            else:
                return jsonify({'error': 'Action invalide'}), 400
            app.vectorizers[lang], app.tfidf_matrices[lang] = create_tfidf_vectors(app.questions[lang], language=lang)
            return jsonify({'status': 'success', 'message': f'Suggestion {action}ée'})
        except Exception as e:
            print(f"Erreur dans /api/validate: {str(e)}")
            return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

    print(app.url_map)
    return app

app = create_app()

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)