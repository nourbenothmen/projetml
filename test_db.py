from app import app, db, Question, Answer

def test_database():
    with app.app_context():
        try:
            # Test de connexion
            db.engine.connect()
            print("✓ Connexion à la base de données réussie!")
            
            # Test de création d'enregistrement
            q = Question(text="Test question", category="test")
            db.session.add(q)
            db.session.commit()
            print("✓ Création d'une question réussie!")
            
            return True
        except Exception as e:
            print(f"✗ Erreur: {e}")
            return False

if __name__ == '__main__':
    test_database()