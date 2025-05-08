import csv
import requests
import re
from typing import Dict, Any

def get_token() -> str:
    """Obtenir un token JWT pour les tests."""
    register_url = 'http://localhost:5000/api/register'
    login_url = 'http://localhost:5000/api/login'
    user_data = {'username': 'nour', 'password': 'nour782002'}
    
    try:
        response = requests.post(register_url, json=user_data)
        response.raise_for_status()
        print("Inscription réussie.")
    except requests.exceptions.HTTPError as e:
        print(f"Erreur lors de l'inscription : {e}")
        print("Utilisateur déjà existant, passage à la connexion...")
    
    try:
        response = requests.post(login_url, json=user_data)
        response.raise_for_status()
        try:
            response_json = response.json()
            token = response_json.get('token')
            if not token:
                raise ValueError("Aucun token trouvé dans la réponse de connexion.")
            print(f"Connexion réussie, token: {token}")
            return token
        except ValueError as ve:
            print(f"Erreur de parsing JSON : {ve}")
            return ''
    except requests.exceptions.HTTPError as e:
        print(f"Erreur lors de la connexion : {e}")
        return ''

def ask_question(question, language, token):
    try:
        response = requests.post(
            'http://localhost:5000/api/ask',
            json={'question': question, 'language': language},
            headers={'Authorization': f'Bearer {token}'}
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Erreur pour la question '{question}' : {response.status_code} {response.text}")
            return None
    except Exception as e:
        print(f"Erreur pour la question '{question}' : {str(e)}")
        return None



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
    token = get_token()
    if not token:
        print("Échec de l'obtention du token. Arrêt des tests.")
        return
    
    correct_answers = 0
    correct_urls = 0
    correct_categories = 0
    total = 0
    similarity_scores = []
    
    with open(test_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row['question']
            expected_answer = row['expected_answer']
            expected_url = row.get('expected_url', '')
            expected_category = row.get('expected_category', '')
            language = row['language']
            total += 1
            
            try:
                response = ask_question(question, language, token)
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
                print(f"Erreur pour la question '{question}' : {e}\n")
    
    answer_accuracy = (correct_answers / total) * 100 if total > 0 else 0
    url_accuracy = (correct_urls / total) * 100 if total > 0 else 0
    category_accuracy = (correct_categories / total) * 100 if total > 0 else 0
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    print(f"Précision des réponses: {answer_accuracy:.2f}% ({correct_answers}/{total})")
    print(f"Précision des URLs: {url_accuracy:.2f}% ({correct_urls}/{total})")
    print(f"Précision des catégories: {category_accuracy:.2f}% ({correct_categories}/{total})")
    print(f"Score de similarité moyen: {avg_similarity:.4f}")

if __name__ == '__main__':
    evaluate_chatbot('test_questions.csv')