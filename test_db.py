from app import app, db, User

def test_database():
    with app.app_context():
        try:
            # Test de connexion
            db.engine.connect()
            print("✓ Connexion à la base de données réussie!")
        
        except Exception as e:
            print(f"✗ Erreur: {e}")
            return False

if __name__ == '__main__':
    test_database()