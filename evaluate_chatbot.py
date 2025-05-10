import csv
import requests
import re
from typing import Dict, Any
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_token() -> str:
    """Obtenir un token JWT pour les tests."""
    register_url = 'http://localhost:5000/api/register'
    login_url = 'http://localhost:5000/api/login'
    user_data = {'username': 'nour', 'password': 'nour782002'}
    
    try:
        response = requests.post(register_url, json=user_data)
        response.raise_for_status()
        logger.info("Inscription réussie.")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Erreur lors de l'inscription : {e}")
        logger.info("Utilisateur déjà existant, passage à la connexion...")
    
    try:
        response = requests.post(login_url, json=user_data)
        response.raise_for_status()
        try:
            response_json = response.json()
            token = response_json.get('token')
            if not token:
                raise ValueError("Aucun token trouvé dans la réponse de connexion.")
            logger.info(f"Connexion réussie, token: {token}")
            return token
        except ValueError as ve:
            logger.error(f"Erreur de parsing JSON : {ve}")
            return ''
    except requests.exceptions.HTTPError as e:
        logger.error(f"Erreur lors de la connexion : {e}")
        return ''

def ask_question(question: str, language: str, token: str) -> Dict[str, Any]:
    """Envoyer une question au chatbot et récupérer la réponse."""
    try:
        response = requests.post(
            'http://localhost:5000/api/ask',
            json={'question': question, 'language': language},
            headers={'Authorization': f'Bearer {token}'}
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Erreur pour la question '{question}' : {response.status_code} {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Erreur pour la question '{question}' : {str(e)}")
        return {}

def normalize_answer(answer: str) -> str:
    """Normaliser la réponse pour la comparaison."""
    if not answer:
        return ''
    answer = re.sub(r'\s+', ' ', answer.lower().strip())
    answer = re.sub(r'note:.*?(?=(consultez|check|suivez|désolé))', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'consultez notre site pour plus d\'informations\.*', '', answer, flags=re.IGNORECASE)
    return answer.strip()

def evaluate_chatbot(test_file: str) -> None:
    """Évaluer le chatbot en comparant les réponses, URLs et catégories aux valeurs attendues."""
    logger.info(f"Chargement du fichier de test: {test_file}")
    
    token = get_token()
    if not token:
        logger.error("Échec de l'obtention du token. Arrêt des tests.")
        return
    
    correct_answers = 0
    correct_urls = 0
    correct_categories = 0
    total = 0
    similarity_scores = []
    valid_rows = []
    
    try:
        with open(test_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Vérifier les en-têtes
            required_fields = ['question', 'expected_answer', 'expected_url', 'expected_category', 'language']
            if not all(field in reader.fieldnames for field in required_fields):
                logger.error(f"En-têtes manquants dans {test_file}. Requis: {required_fields}, Trouvés: {reader.fieldnames}")
                return
            
            for row_idx, row in enumerate(reader):
                question = row['question'].strip()
                expected_answer = row['expected_answer'].strip()
                expected_url = row.get('expected_url', '').strip()
                expected_category = row.get('expected_category', '').strip()
                language = row['language'].strip()
                
                # Vérifier si la ligne est valide
                if not question or not expected_answer or not language:
                    logger.warning(f"Ligne {row_idx + 2} ignorée: Données manquantes - {row}")
                    continue
                
                valid_rows.append(row)
                total += 1
                logger.debug(f"Ligne {row_idx + 2}: {row}")
                
                try:
                    response = ask_question(question, language, token)
                    if not response:
                        logger.error(f"Aucune réponse reçue pour la question '{question}'")
                        continue
                    
                    actual_answer = response.get('answer', '')
                    actual_url = response.get('url', '')
                    actual_category = response.get('category', '')
                    similarity_score = response.get('similarity_score', 0.0)
                    similarity_scores.append(similarity_score)
                    
                    normalized_expected = normalize_answer(expected_answer)
                    normalized_actual = normalize_answer(actual_answer)
                    
                    is_answer_correct = normalized_actual == normalized_expected
                    is_url_correct = actual_url == expected_url
                    is_category_correct = actual_category == expected_category
                    
                    if is_answer_correct:
                        correct_answers += 1
                    if is_url_correct:
                        correct_urls += 1
                    if is_category_correct:
                        correct_categories += 1
                    
                    print(f"Question: {question}")
                    print(f"Expected Answer (normalized): {normalized_expected}")
                    print(f"Got Answer (normalized): {normalized_actual}")
                    print(f"Expected URL: {expected_url}")
                    print(f"Got URL: {actual_url}")
                    print(f"Expected Category: {expected_category}")
                    print(f"Got Category: {actual_category}")
                    print(f"Similarity Score: {similarity_score:.4f}")
                    print(f"Answer Correct: {is_answer_correct}")
                    print(f"URL Correct: {is_url_correct}")
                    print(f"Category Correct: {is_category_correct}\n")
                except Exception as e:
                    logger.error(f"Erreur pour la question '{question}' : {e}")
                    continue
        
        logger.info(f"Nombre total de questions valides chargées: {total}")
        
        answer_accuracy = (correct_answers / total) * 100 if total > 0 else 0
        url_accuracy = (correct_urls / total) * 100 if total > 0 else 0
        category_accuracy = (correct_categories / total) * 100 if total > 0 else 0
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        print(f"Précision des réponses: {answer_accuracy:.2f}% ({correct_answers}/{total})")
        print(f"Précision des URLs: {url_accuracy:.2f}% ({correct_urls}/{total})")
        print(f"Précision des catégories: {category_accuracy:.2f}% ({correct_categories}/{total})")
        print(f"Score de similarité moyen: {avg_similarity:.4f}")
        
    except FileNotFoundError:
        logger.error(f"Le fichier {test_file} n'existe pas.")
    except UnicodeDecodeError:
        logger.error(f"Erreur d'encodage dans {test_file}. Assurez-vous qu'il est en UTF-8 sans BOM.")
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {str(e)}")

if __name__ == '__main__':
    evaluate_chatbot('test_questions.csv')