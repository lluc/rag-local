#!/usr/bin/env python3
"""
Module de gestion et téléchargement de modèles LLM
Inspiré du système de téléchargement automatique
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

class LLMDownloader:
    """Gestionnaire de téléchargement et gestion des modèles LLM"""

    # Catalogue des modèles disponibles
    MODELS = {
        "llama-3.2-3b": {
            "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "size": "~1.9GB",
            "description": "Llama 3.2 3B (Meta, RECOMMANDÉ - équilibré)",
            "performance": "excellent",
            "memory": "4GB+"
        },
        "mistral-7b-instruct-q4": {
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "size": "~4.4GB",
            "description": "Mistral 7B v0.2 (Mistral AI, très performant)",
            "performance": "excellent",
            "memory": "8GB+"
        },
        "phi-3-mini": {
            "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
            "filename": "phi-3-mini-4k-instruct-q4.gguf",
            "size": "~2.4GB",
            "description": "Phi-3 Mini (Microsoft, compact et rapide)",
            "performance": "bon",
            "memory": "4GB+"
        },
        "gemma-2-2b": {
            "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
            "filename": "gemma-2-2b-it-Q4_K_M.gguf",
            "size": "~1.7GB",
            "description": "Gemma 2 2B (Google, très léger)",
            "performance": "bon",
            "memory": "3GB+"
        },
        "tinyllama": {
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size": "~700MB",
            "description": "TinyLlama 1.1B (ultra-rapide, tests)",
            "performance": "basique",
            "memory": "2GB+"
        }
    }

    def __init__(self, models_dir: str = "./models"):
        """
        Initialise le gestionnaire de modèles

        Args:
            models_dir: Répertoire de stockage des modèles
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def list_available_models(self) -> None:
        """Affiche la liste des modèles disponibles au téléchargement"""
        print("\n📦 Modèles LLM disponibles:")
        print("=" * 60)

        for key, model in self.MODELS.items():
            status = "✅ Installé" if self.is_model_installed(key) else "📥 Disponible"
            print(f"\n🔹 {key}")
            print(f"   {model['description']}")
            print(f"   Taille: {model['size']} | Performance: {model['performance']} | RAM: {model['memory']}")
            print(f"   Status: {status}")

    def list_installed_models(self) -> Dict[str, str]:
        """
        Liste les modèles installés localement

        Returns:
            Dict mapping model_key -> file_path
        """
        installed = {}

        for key, model in self.MODELS.items():
            filepath = self.models_dir / model["filename"]
            if filepath.exists() and filepath.stat().st_size > 1024 * 1024:  # > 1MB
                installed[key] = str(filepath)

        return installed

    def is_model_installed(self, model_key: str) -> bool:
        """Vérifie si un modèle est installé"""
        if model_key not in self.MODELS:
            return False

        filepath = self.models_dir / self.MODELS[model_key]["filename"]
        return filepath.exists() and filepath.stat().st_size > 1024 * 1024

    def get_model_path(self, model_key: str) -> Optional[str]:
        """
        Récupère le chemin d'un modèle installé

        Args:
            model_key: Clé du modèle

        Returns:
            Chemin du fichier ou None si non installé
        """
        if not self.is_model_installed(model_key):
            return None

        return str(self.models_dir / self.MODELS[model_key]["filename"])

    def check_disk_space(self, size_str: str) -> bool:
        """
        Vérifie l'espace disque disponible

        Args:
            size_str: Taille du modèle (ex: "~1.9GB")

        Returns:
            True si assez d'espace disponible
        """
        try:
            # Parser la taille
            size_clean = size_str.replace('~', '').replace('GB', '').replace('MB', '')
            size_gb = float(size_clean)
            if 'MB' in size_str:
                size_gb /= 1024

            # Vérifier l'espace libre
            statvfs = os.statvfs(self.models_dir)
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)

            if free_gb < size_gb + 1:  # +1GB de marge
                print(f"⚠️  Espace disque faible: {free_gb:.1f}GB libre, {size_gb:.1f}GB nécessaires")
                return False

            return True
        except Exception:
            return True  # En cas d'erreur, on continue

    def download_with_curl(self, url: str, filepath: Path) -> bool:
        """
        Télécharge un fichier avec curl

        Args:
            url: URL de téléchargement
            filepath: Chemin de destination

        Returns:
            True si succès, False sinon
        """
        # Vérifier que curl est disponible
        if not shutil.which("curl"):
            print("❌ curl n'est pas installé. Installation requise:")
            print("   sudo apt install curl  # Linux")
            print("   brew install curl      # macOS")
            return False

        print(f"📥 Téléchargement en cours...")
        print(f"📂 Destination: {filepath}")

        try:
            cmd = [
                "curl",
                "-L",                    # Suivre les redirections
                "-C", "-",              # Reprendre téléchargement si interrompu
                "-o", str(filepath),    # Fichier de sortie
                "--progress-bar",       # Barre de progression
                "--fail",               # Échouer sur erreurs HTTP
                url
            ]

            subprocess.run(cmd, check=True)

            # Vérifier que le fichier est valide
            if filepath.stat().st_size < 1024 * 1024:  # < 1MB
                print("❌ Fichier téléchargé trop petit, probablement corrompu")
                filepath.unlink()
                return False

            print(f"\n✅ Téléchargement terminé!")
            print(f"📊 Taille: {filepath.stat().st_size / (1024**2):.1f} MB")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Erreur de téléchargement (code {e.returncode})")
            if filepath.exists():
                filepath.unlink()
            return False

        except KeyboardInterrupt:
            print(f"\n⚠️  Téléchargement interrompu par l'utilisateur")
            if filepath.exists():
                filepath.unlink()
            return False

    def download_model(self, model_key: str, force: bool = False) -> bool:
        """
        Télécharge un modèle

        Args:
            model_key: Clé du modèle à télécharger
            force: Forcer le téléchargement même si déjà présent

        Returns:
            True si succès, False sinon
        """
        if model_key not in self.MODELS:
            print(f"❌ Modèle '{model_key}' non trouvé dans le catalogue")
            print(f"Modèles disponibles: {list(self.MODELS.keys())}")
            return False

        model = self.MODELS[model_key]
        filepath = self.models_dir / model["filename"]

        # Vérifier si déjà installé
        if self.is_model_installed(model_key) and not force:
            print(f"✅ Le modèle '{model_key}' est déjà installé")
            print(f"📂 Emplacement: {filepath}")
            return True

        # Vérifier l'espace disque
        if not self.check_disk_space(model["size"]):
            response = input("Continuer malgré l'espace faible? (y/N): ").strip().lower()
            if response != 'y':
                print("Téléchargement annulé")
                return False

        print(f"\n🚀 Téléchargement de {model['description']}")
        print(f"📊 Taille: {model['size']}")

        # Télécharger
        success = self.download_with_curl(model["url"], filepath)

        if success:
            print(f"\n🎉 Modèle '{model_key}' installé avec succès!")
            return True
        else:
            print(f"\n❌ Échec du téléchargement de '{model_key}'")
            print("\n💡 Solutions alternatives:")
            print("   1. Réessayer plus tard")
            print("   2. Télécharger manuellement:")
            print(f"      URL: {model['url']}")
            print(f"      Destination: {filepath}")
            return False

    def remove_model(self, model_key: str) -> bool:
        """
        Supprime un modèle installé

        Args:
            model_key: Clé du modèle à supprimer

        Returns:
            True si succès, False sinon
        """
        if not self.is_model_installed(model_key):
            print(f"❌ Modèle '{model_key}' non installé")
            return False

        filepath = self.models_dir / self.MODELS[model_key]["filename"]

        try:
            filepath.unlink()
            print(f"✅ Modèle '{model_key}' supprimé")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la suppression: {e}")
            return False

    def get_recommended_model(self) -> str:
        """Retourne la clé du modèle recommandé"""
        return "llama-3.2-3b"

    def auto_download_recommended(self) -> Optional[str]:
        """
        Télécharge automatiquement le modèle recommandé s'il n'existe aucun modèle

        Returns:
            Chemin du modèle téléchargé ou None si échec
        """
        installed = self.list_installed_models()

        if installed:
            # Un modèle est déjà installé
            return list(installed.values())[0]

        # Aucun modèle installé, télécharger le recommandé
        recommended = self.get_recommended_model()
        print(f"📥 Téléchargement automatique du modèle recommandé: {recommended}")

        if self.download_model(recommended):
            return self.get_model_path(recommended)

        return None


# Exemple d'utilisation
if __name__ == "__main__":
    downloader = LLMDownloader()

    # Lister les modèles disponibles
    downloader.list_available_models()

    print("\n" + "="*60)

    # Lister les modèles installés
    installed = downloader.list_installed_models()
    if installed:
        print(f"\n✅ Modèles installés: {len(installed)}")
        for key, path in installed.items():
            print(f"   🔹 {key}: {path}")
    else:
        print("\n📭 Aucun modèle installé")