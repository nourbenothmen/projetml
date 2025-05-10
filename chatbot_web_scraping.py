import requests
from bs4 import BeautifulSoup
import csv

# Fonction pour nettoyer le texte
def clean_text(text):
    return ' '.join(text.split()).strip()

# URL du sitemap
sitemap_url = "https://isetsf.rnu.tn/fr/sitemap"
base_url = "https://isetsf.rnu.tn"

# En-tête CSV
csv_headers = ["question", "answer", "category", "language", "url", "intent"]

# Récupération du contenu du sitemap
response = requests.get(sitemap_url)
soup = BeautifulSoup(response.content, "html.parser")

# Récupération des liens internes
links = soup.select(".sitemap a")
data = []

for link in links:
    href = link.get("href")

    # Ignorer les liens vides ou javascript
    if not href or href.startswith("javascript") or href.strip() == "#":
        continue

    # Compléter l'URL si nécessaire
    full_url = base_url + href if href.startswith("/") else href

    try:
        page = requests.get(full_url)
        page_soup = BeautifulSoup(page.content, "html.parser")

        # Récupérer le titre et le contenu principal
        title = page_soup.title.string if page_soup.title else "Informations sur l'ISET de Sfax"
        content = page_soup.get_text()
        content = clean_text(content)

        # Ne garder que les pages avec un minimum de contenu
        if len(content) < 200:
            continue

        # Génération automatique de question/réponse
        question = f"Que faut-il savoir sur : {title} ?"
        answer = content[:500] + "..." if len(content) > 500 else content

        data.append([
            question,
            answer,
            "iset-sfax",
            "fr",
            full_url,
            "faq"
        ])

    except Exception as e:
        print(f"Erreur lors du traitement de {full_url}: {e}")

# Écriture dans un fichier CSV
with open("iset_sfax_qna.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(data)

print("✅ Fichier iset_sfax_qna.csv généré avec succès.")
