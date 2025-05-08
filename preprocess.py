import spacy
import string
from nltk.corpus import stopwords
import nltk
import re

# Télécharger les ressources NLTK nécessaires
nltk.download('stopwords')

# Charger le modèle français de spaCy
try:
    nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])
except OSError:
    raise Exception("Modèle spaCy 'fr_core_news_sm' non installé. Exécutez : python -m spacy download fr_core_news_sm")

def preprocess_text(text, language='fr'):
    """
    Prétraiter le texte : tokenisation, suppression des mots vides, lemmatisation.
    Args:
        text (str): Texte à prétraiter.
        language (str): Langue du texte ('fr' ou 'en').
    Returns:
        str: Texte prétraité.
    """
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les caractères spéciaux (garder les lettres et espaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Traiter le texte avec spaCy
    doc = nlp(text)
    
    # Supprimer les mots vides et la ponctuation, lemmatiser
    stop_words = set(stopwords.words('french' if language == 'fr' else 'english'))
    if language == 'fr':
        stop_words = stop_words.union({'quelles', 'quel', 'quels', 'le', 'la', 'l', 'de', 'à', 'd', 'il', 'des'})
    else:
        stop_words = stop_words.union({'a', 'an', 'the', 'is', 'are', 'what'})
    
    tokens = []
    for token in doc:
        # Conserver certains mots clés non lemmatisés (ex. "sfax", "iset")
        if token.text.lower() in ['sfax', 'iset']:
            tokens.append(token.text.lower())
        elif token.text not in stop_words and token.text not in string.punctuation and token.is_alpha:
            # Normaliser certains termes clés
            lemma = token.lemma_.lower()
            if lemma in ['critère', 'condition']:
                tokens.append('condition')
            else:
                tokens.append(lemma)
    
    # Rejoindre les tokens en une chaîne
    return ' '.join(tokens)