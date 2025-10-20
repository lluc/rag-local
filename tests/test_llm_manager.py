"""
Script de test pour le LLMManager
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path pour importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager import LLMManager


def test_model_check():
    """Test de vérification de modèle"""
    print("="*60)
    print("TEST 1: Vérification de modèle existant")
    print("="*60)

    manager = LLMManager()

    # Test avec un modèle qui n'existe pas
    exists = manager.check_model_exists("./models/non_existent.gguf")
    print(f"Résultat: {exists}")
    assert not exists, "Le modèle ne devrait pas exister"
    print("✓ Test passé\n")


def test_list_models():
    """Test de listage des modèles"""
    print("="*60)
    print("TEST 2: Listage des modèles locaux")
    print("="*60)

    manager = LLMManager()
    models = manager.list_available_models()
    print(f"Nombre de modèles trouvés: {len(models)}")
    for model in models:
        print(f"  - {model}")
    print("✓ Test passé\n")


def test_recommended_models():
    """Test des modèles recommandés"""
    print("="*60)
    print("TEST 3: Modèles recommandés")
    print("="*60)

    manager = LLMManager()
    recommended = manager.get_recommended_models()
    print(f"Nombre de modèles recommandés: {len(recommended)}")
    assert len(recommended) > 0, "Il devrait y avoir des modèles recommandés"

    for key, info in recommended.items():
        print(f"\n{key}:")
        print(f"  Description: {info['description']}")
        print(f"  Taille: {info['size']}")
    print("\n✓ Test passé\n")


def test_get_model_no_download():
    """Test de récupération sans téléchargement automatique"""
    print("="*60)
    print("TEST 4: Récupération sans téléchargement auto")
    print("="*60)

    manager = LLMManager()

    try:
        # Essayer de récupérer sans téléchargement automatique
        model_path = manager.get_or_download_model(
            model_path="./models/non_existent.gguf",
            auto_download=False
        )
        print(f"❌ Le test devrait échouer mais a retourné: {model_path}")
    except FileNotFoundError as e:
        print(f"✓ Exception attendue capturée: {e}")
        print("✓ Test passé\n")


if __name__ == "__main__":
    print("\n🧪 TESTS DU LLM MANAGER\n")

    try:
        test_model_check()
        test_list_models()
        test_recommended_models()
        test_get_model_no_download()

        print("="*60)
        print("✅ TOUS LES TESTS ONT RÉUSSI")
        print("="*60)

    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
