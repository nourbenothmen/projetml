from app import create_app
from extensions import db
from models import User

app = create_app()

with app.app_context():
    #print("Modèles détectés :", [cls.__name__ for cls in db.Model.registry._class_registry.values()])
    db.drop_all()
    db.create_all()
    print("Base de données initialisée avec des données initiales.")