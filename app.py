from flask import Flask, request, jsonify
from flask_cors import CORS
from extensions import db
from models import User
from text_representation import create_tfidf_vectors, preprocess_text
from chatbot_model import find_best_match
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import csv
import jwt
import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import re
import os
from typing import Dict, List, Tuple, Optional

def load_data_from_csv(app_instance):
    expected_fieldnames = ['question', 'answer', 'category', 'language', 'url', 'intent']
    csv_path = app_instance.config['CSV_PATH']
    
    # Vérifier que le fichier CSV existe
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas.")
    
    # Initialiser les structures de données
    app_instance.questions = {'fr': [], 'en': []}
    app_instance.answers = {'fr': [], 'en': []}
    seen_questions = set()  # Pour éviter les duplications
    
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Vérifier les en-têtes
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
                    
                    if not question or not answer or not category or not lang:
                        print(f"Ignoré: Ligne incomplète - {row}")
                        continue
                    
                    if lang not in ['fr', 'en']:
                        print(f"Ignoré: Langue invalide '{lang}' pour question: {question}")
                        continue
                    
                    if question not in seen_questions:
                        seen_questions.add(question)
                        app_instance.questions[lang].append(question)
                        app_instance.answers[lang].append((answer, url, len(app_instance.answers[lang])))
                        print(f"Chargé: {question} (lang: {lang}, category: {category}, url: {url})")
                    else:
                        print(f"Ignoré: Question en double: {question}")
                except KeyError as e:
                    print(f"Erreur: Clé manquante {e} dans la ligne: {row}")
                    continue
                except Exception as e:
                    print(f"Erreur inattendue dans la ligne: {row}, erreur: {str(e)}")
                    continue
    
        print(f"Total questions chargées: fr={len(app_instance.questions['fr'])}, en={len(app_instance.questions['en'])}")
        print(f"Questions fr: {app_instance.questions['fr']}")
        print(f"Questions en: {app_instance.questions['en']}")
    
    except UnicodeDecodeError:
        print(f"Erreur: Problème d'encodage dans {csv_path}. Assurez-vous qu'il est en UTF-8 sans BOM.")
        raise
    except Exception as e:
        print(f"Erreur lors du chargement de {csv_path}: {str(e)}")
        raise
    
    # Créer les vecteurs TF-IDF
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
            print(f"Nombre de questions: {len(questions)}")
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

def save_to_csv(question: str, answer: str, category: str, language: str, url: str = '', intent: str = ''):
    try:
        rows = []
        with open(app.config['CSV_PATH'], newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        rows.append({'question': question, 'answer': answer, 'category': category, 'language': language, 'url': url, 'intent': intent})
        with open(app.config['CSV_PATH'], 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['question', 'answer', 'category', 'language', 'url', 'intent'])
            writer.writeheader()
            writer.writerows(rows)
        lang = language
        app.questions[lang].append(question)
        app.answers[lang].append((answer, url, len(app.answers[lang])))
        app.vectorizers[lang], app.tfidf_matrices[lang] = create_tfidf_vectors(app.questions[lang], language=lang)
        print(f"Sauvegardé: {question} (lang: {lang}, category: {category})")
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

    try:
        load_data_from_csv(app)
        train_category_classifier(app)
    except Exception as e:
        print(f"Erreur lors de l'initialisation de l'application: {str(e)}")
        raise

    print(f"Questions chargées: {app.questions}")
    print(f"Réponses chargées: {app.answers}")

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
    @token_required
    def ask_question(current_user):
        try:
            data = request.get_json()
            print("Données reçues:", data)
            if not data or 'question' not in data:
                return jsonify({'error': 'Question requise'}), 400
            question_text = data.get('question')
            preferred_lang = data.get('language', 'fr')
            print(f"Question reçue: {question_text}, Langue préférée: {preferred_lang}")
            if not isinstance(question_text, str):
                return jsonify({'error': f'Question doit être une chaîne, reçu: {type(question_text)}'}), 400

            try:
                detected_lang = detect(question_text)
            except Exception as e:
                print(f"Erreur lors de la détection de langue: {e}")
                detected_lang = 'fr'
            question_lang = 'en' if detected_lang.startswith('en') else 'fr'
            if preferred_lang in ['fr', 'en'] and preferred_lang != question_lang:
                question_lang = preferred_lang
            print(f"Langue détectée: {detected_lang}, Langue finale: {question_lang}")

            processed_query = preprocess_text(question_text, language=question_lang)
            print(f"Query prétraitée: {processed_query}")
            
            predicted_category = None
            if app.category_classifier and app.category_vectorizer:
                predicted_category = predict_category(processed_query, app.category_vectorizer, app.category_classifier, question_lang)
                print(f"Catégorie prédite: {predicted_category}")
            
            predicted_intent = (app.intent_classifier.predict(app.intent_vectorizer.transform([processed_query]))[0]
                              if app.intent_classifier and app.intent_vectorizer else None)
            print(f"Intention prédite: {predicted_intent}")
            entities = {}
            courses = r'\b(Génie Civil|Informatique|Génie Mécanique|Programmation|Algorithmes)\b'
            if re.search(courses, question_text, re.IGNORECASE):
                entities['course'] = re.search(courses, question_text, re.IGNORECASE).group()
            print(f"Entités détectées: {entities}")

            questions_for_lang = app.questions.get(question_lang, [])
            print(f"Questions disponibles pour {question_lang}: {questions_for_lang}")
            if not questions_for_lang or not app.vectorizers.get(question_lang):
                return jsonify({'error': f'Aucune question disponible en {question_lang}'}), 500

            best_match_idx, similarity_score = find_best_match(
                processed_query, questions_for_lang, app.vectorizers[question_lang], app.tfidf_matrices[question_lang], question_lang
            ) if questions_for_lang else (-1, 0.0)
            print(f"Meilleur index: {best_match_idx}, Score de similarité: {similarity_score}")

            app.similarity_threshold = adjust_similarity_threshold()
            print(f"Seuil de similarité: {app.similarity_threshold}")
            if similarity_score < app.similarity_threshold:
                return jsonify({
                    'answer': 'Désolé, je n’ai pas de réponse pertinente.',
                    'intent': predicted_intent,
                    'entities': entities,
                    'language': question_lang,
                    'category': predicted_category or 'Inconnu',
                    'similarity_score': round(similarity_score, 4)
                }), 200

            if best_match_idx < 0 or best_match_idx >= len(questions_for_lang):
                return jsonify({
                    'answer': 'Aucune correspondance trouvée.',
                    'intent': predicted_intent,
                    'entities': entities,
                    'language': question_lang,
                    'category': predicted_category or 'Inconnu'
                }), 404

            matched_question = questions_for_lang[best_match_idx]
            print(f"Question correspondante: {matched_question}")
            lang = preferred_lang if preferred_lang in ['fr', 'en'] else question_lang
            print(f"Langue utilisée pour la réponse: {lang}")
            answer_text, answer_url, answer_id = app.answers[lang][best_match_idx]
            print(f"Réponse: {answer_text}, URL: {answer_url}, ID: {answer_id}")
            if preferred_lang and preferred_lang != lang:
                note = ("Note: No answer in English, using French." if preferred_lang == 'en' else
                        "Note: Pas de réponse en français, utilisant l'anglais.")
                answer_text = f"{note}\n{answer_text}"

            if 'course' in entities and not answer_url:
                answer_url = f"https://isetsf.rnu.tn/fr/formation/{entities['course'].lower().replace(' ', '-')}"
                answer_text += f" Consultez notre site pour plus d'informations."
                
            append_feedback(question_text, answer_id, 'pending', similarity_score)

            with open(app.config['CSV_PATH'], newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                category = None
                for row in reader:
                    if row['question'] == matched_question and row['language'] == lang:
                        category = row['category']
                        print(f"Catégorie trouvée dans CSV pour '{matched_question}': {category}")
                        break
                if not category:
                    print(f"Aucune catégorie trouvée pour question: {matched_question}, langue: {lang}")
                    category = predicted_category or 'Inconnu'

            return jsonify({
                'answer': answer_text,
                'url': answer_url,
                'category': category,
                'similarity_score': round(similarity_score, 4),
                'answer_id': answer_id,
                'language': lang,
                'intent': predicted_intent,
                'entities': entities
            })
        except Exception as e:
            print(f"Erreur dans /api/ask: {str(e)}")
            return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

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