from preprocess import preprocess_text

# Exemples de textes
texts = [
    "Quelles sont les conditions d'admission à l'ISET Sfax ?",
    "Quels sont les horaires de la bibliothèque ?",
    "L'ISET propose-t-il des cours d'informatique ?"
]

for text in texts:
    processed_text = preprocess_text(text)
    print(f"Texte original : {text}")
    print(f"Texte prétraité : {processed_text}\n")