#!/usr/bin/env python3
"""
Test de sélection de modèle pour vérifier que Llama-3.2-3B est choisi par défaut
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire courant au path pour les imports
sys.path.insert(0, os.getcwd())

def test_model_selection():
    """Test de sélection de modèle sans dépendances lourdes"""

    try:
        from llm_manager import LLMManager

        # Créer le gestionnaire
        manager = LLMManager(models_dir="./models")

        print("🧪 Test de sélection de modèle")
        print("=" * 50)

        # Test 1: Vérifier que le modèle Llama-3.2-3B existe
        llama_path = manager.models_dir / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        if llama_path.exists():
            print(f"✅ Llama-3.2-3B trouvé: {llama_path}")
        else:
            print(f"❌ Llama-3.2-3B NON trouvé: {llama_path}")
            return False

        # Test 2: Vérifier que la clé existe dans RECOMMENDED_MODELS
        if "llama-3.2-3b" in manager.RECOMMENDED_MODELS:
            print(f"✅ Clé 'llama-3.2-3b' trouvée dans RECOMMENDED_MODELS")
            filename = manager.RECOMMENDED_MODELS["llama-3.2-3b"]["filename"]
            print(f"   Nom de fichier attendu: {filename}")
        else:
            print(f"❌ Clé 'llama-3.2-3b' NON trouvée dans RECOMMENDED_MODELS")
            return False

        # Test 3: Simuler get_or_download_model (sans auto_download pour éviter l'interaction)
        print(f"\n🔍 Test de get_or_download_model avec model_key='llama-3.2-3b'")

        # Créer une version modifiée sans interaction utilisateur
        try:
            result = manager.get_or_download_model(
                model_path=None,
                model_key="llama-3.2-3b",
                auto_download=False  # Pas de téléchargement automatique
            )
            print(f"✅ Modèle sélectionné: {result}")

            # Vérifier que c'est bien le bon modèle
            if "Llama-3.2-3B-Instruct-Q4_K_M.gguf" in result:
                print(f"✅ SUCCESS: Le bon modèle Llama-3.2-3B a été sélectionné!")
                return True
            else:
                print(f"❌ ERREUR: Mauvais modèle sélectionné (attendu: Llama-3.2-3B)")
                return False

        except Exception as e:
            print(f"❌ Erreur lors de la sélection: {e}")
            return False

    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False

if __name__ == "__main__":
    success = test_model_selection()
    if success:
        print(f"\n🎉 Tous les tests réussis! Llama-3.2-3B sera utilisé par défaut.")
        sys.exit(0)
    else:
        print(f"\n💥 Tests échoués! Le modèle par défaut pourrait ne pas être correct.")
        sys.exit(1)