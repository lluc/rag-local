"""
LLM Manager - Gestion des modèles de langage locaux (GGUF)
Vérification, téléchargement et chargement automatiques
"""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import requests
from tqdm import tqdm


class LLMManager:
    """Gestionnaire de modèles LLM locaux avec téléchargement automatique"""

    # Modèles recommandés avec leurs URLs Hugging Face
    RECOMMENDED_MODELS = {
        "mistral-7b-instruct-q4": {
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "size": "4.37 GB",
            "description": "Mistral 7B Instruct - Quantization Q4_K_M (recommandé)"
        },
        "mistral-7b-instruct-q5": {
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
            "filename": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
            "size": "5.13 GB",
            "description": "Mistral 7B Instruct - Quantization Q5_K_M (meilleure qualité)"
        },
        "llama-3.2-3b-q4": {
            "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "size": "1.95 GB",
            "description": "Llama 3.2 3B Instruct - Plus léger"
        },
        "phi-3-mini-q4": {
            "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
            "filename": "Phi-3-mini-4k-instruct-q4.gguf",
            "size": "2.23 GB",
            "description": "Phi-3 Mini - Très performant et léger"
        }
    }

    def __init__(self, models_dir: str = "./models"):
        """
        Initialise le gestionnaire de modèles

        Args:
            models_dir: Dossier de stockage des modèles
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def check_model_exists(self, model_path: str) -> bool:
        """
        Vérifie si un modèle existe localement

        Args:
            model_path: Chemin vers le modèle

        Returns:
            True si le modèle existe, False sinon
        """
        path = Path(model_path)
        if path.exists() and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ Modèle trouvé: {model_path} ({size_mb:.2f} MB)")
            return True
        return False

    def list_available_models(self) -> list[str]:
        """Liste tous les modèles GGUF disponibles localement"""
        models = list(self.models_dir.glob("*.gguf"))
        return [str(m) for m in models]

    def get_recommended_models(self) -> dict:
        """Retourne la liste des modèles recommandés"""
        return self.RECOMMENDED_MODELS

    def download_model(
        self,
        url: str,
        filename: Optional[str] = None,
        chunk_size: int = 8192
    ) -> str:
        """
        Télécharge un modèle depuis une URL

        Args:
            url: URL du modèle à télécharger
            filename: Nom du fichier de destination (déduit de l'URL si None)
            chunk_size: Taille des chunks pour le téléchargement

        Returns:
            Chemin vers le modèle téléchargé
        """
        # Déterminer le nom de fichier
        if filename is None:
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name

        output_path = self.models_dir / filename

        # Vérifier si déjà téléchargé
        if output_path.exists():
            print(f"✓ Le modèle existe déjà: {output_path}")
            return str(output_path)

        print(f"📥 Téléchargement du modèle: {filename}")
        print(f"   Source: {url}")

        try:
            # Requête avec stream pour les gros fichiers
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Obtenir la taille totale
            total_size = int(response.headers.get('content-length', 0))

            # Télécharger avec barre de progression
            with open(output_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=filename
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"✓ Modèle téléchargé: {output_path}")
            return str(output_path)

        except requests.exceptions.RequestException as e:
            # Nettoyer en cas d'erreur
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Erreur lors du téléchargement: {e}")

    def download_recommended_model(self, model_key: str) -> str:
        """
        Télécharge un modèle recommandé par sa clé

        Args:
            model_key: Clé du modèle dans RECOMMENDED_MODELS

        Returns:
            Chemin vers le modèle téléchargé
        """
        if model_key not in self.RECOMMENDED_MODELS:
            available = ", ".join(self.RECOMMENDED_MODELS.keys())
            raise ValueError(
                f"Modèle '{model_key}' inconnu. "
                f"Modèles disponibles: {available}"
            )

        model_info = self.RECOMMENDED_MODELS[model_key]
        print(f"📦 {model_info['description']}")
        print(f"📊 Taille: {model_info['size']}")

        return self.download_model(
            url=model_info['url'],
            filename=model_info['filename']
        )

    def get_or_download_model(
        self,
        model_path: Optional[str] = None,
        model_key: str = "mistral-7b-instruct-q4",
        auto_download: bool = True
    ) -> str:
        """
        Obtient un modèle (vérifie l'existence ou télécharge)

        Args:
            model_path: Chemin vers un modèle existant (prioritaire)
            model_key: Clé du modèle recommandé à télécharger si besoin
            auto_download: Télécharger automatiquement si absent

        Returns:
            Chemin vers le modèle prêt à utiliser
        """
        # Si un chemin est fourni, vérifier s'il existe
        if model_path:
            if self.check_model_exists(model_path):
                return model_path
            print(f"⚠️  Modèle non trouvé: {model_path}")

        # Chercher dans le dossier models
        local_models = self.list_available_models()
        if local_models:
            print(f"✓ Modèles locaux trouvés: {len(local_models)}")
            model = local_models[0]
            print(f"   Utilisation de: {model}")
            return model

        # Télécharger automatiquement si activé
        if auto_download:
            print("\n🤖 Aucun modèle local trouvé")
            print(f"   Téléchargement automatique du modèle recommandé: {model_key}")

            user_input = input("\n   Continuer? [O/n]: ").strip().lower()
            if user_input in ['n', 'non', 'no']:
                raise RuntimeError("Téléchargement annulé par l'utilisateur")

            return self.download_recommended_model(model_key)

        # Aucun modèle disponible
        raise FileNotFoundError(
            f"Aucun modèle trouvé dans {self.models_dir}. "
            "Utilisez download_recommended_model() ou fournissez model_path."
        )

    def print_available_models(self):
        """Affiche les modèles disponibles localement et recommandés"""
        print("\n" + "="*60)
        print("📦 MODÈLES LOCAUX")
        print("="*60)

        local_models = self.list_available_models()
        if local_models:
            for i, model in enumerate(local_models, 1):
                path = Path(model)
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"{i}. {path.name} ({size_mb:.2f} MB)")
        else:
            print("Aucun modèle local trouvé")

        print("\n" + "="*60)
        print("🌟 MODÈLES RECOMMANDÉS")
        print("="*60)

        for key, info in self.RECOMMENDED_MODELS.items():
            print(f"\n{key}:")
            print(f"  📝 {info['description']}")
            print(f"  📊 Taille: {info['size']}")
            print(f"  🔗 {info['url']}")


# Exemple d'utilisation
if __name__ == "__main__":
    manager = LLMManager()

    # Afficher les modèles disponibles
    manager.print_available_models()

    # Obtenir ou télécharger un modèle
    # model_path = manager.get_or_download_model(
    #     model_key="mistral-7b-instruct-q4",
    #     auto_download=True
    # )
    # print(f"\n✓ Modèle prêt: {model_path}")
