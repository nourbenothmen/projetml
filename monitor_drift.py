import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset
import os

# Charger les données de référence (chatbot_data.csv)
reference_data = pd.read_csv('chatbot_data.csv')[['question']]

# Charger les données actuelles (production_questions.csv)
if not os.path.exists('production_questions.csv'):
    print("Aucune donnée de production disponible pour le moment.")
    current_data = pd.DataFrame({'question': []})
else:
    current_data = pd.read_csv('production_questions.csv')[['question']]

# Générer un rapport de dérive avec Evidently
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Sauvegarder le rapport
report.save_html('drift_report.html')
print("Rapport de dérive généré: drift_report.html")