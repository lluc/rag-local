"""
CLI pour le système RAG Local
Interface en ligne de commande pour gérer documents et requêtes
"""

import argparse
import sys
from pathlib import Path
from rag_local import LocalRAG
from llm_manager import LLMManager
from llm_downloader import LLMDownloader


def cmd_init(args):
    """Initialise le système RAG avec un modèle"""
    print("🚀 Initialisation du système RAG Local\n")

    try:
        rag = LocalRAG(
            model_path=args.model_path,
            model_key=args.model_key,
            auto_download=args.auto_download,
            n_threads=args.threads,
            persist_directory=args.db_path,
            retriever_k=args.retriever_k,
            chain_type=args.chain_type
        )
        print("\n✅ Système RAG initialisé avec succès !")
        return 0
    except Exception as e:
        print(f"\n❌ Erreur lors de l'initialisation: {e}")
        return 1


def cmd_index(args):
    """Indexe un document ou un dossier"""
    print("📚 Indexation de documents\n")

    try:
        rag = LocalRAG(
            model_path=args.model_path,
            model_key=args.model_key,
            auto_download=False,  # Ne pas télécharger pendant l'indexation
            n_threads=args.threads,
            persist_directory=args.db_path,
            retriever_k=args.retriever_k,
            chain_type=args.chain_type
        )

        path = Path(args.path)

        if not path.exists():
            print(f"❌ Chemin introuvable: {args.path}")
            return 1

        if path.is_file():
            print(f"📄 Indexation du fichier: {path}")
            rag.index_document(str(path))
        elif path.is_dir():
            print(f"📁 Indexation du dossier: {path}")
            rag.index_directory(str(path))
        else:
            print(f"❌ Type de chemin invalide: {args.path}")
            return 1

        print("\n✅ Indexation terminée avec succès !")
        return 0

    except Exception as e:
        print(f"\n❌ Erreur lors de l'indexation: {e}")
        return 1


def cmd_query(args):
    """Interroge le système RAG"""
    print("🔍 Interrogation du système RAG\n")

    try:
        rag = LocalRAG(
            model_path=args.model_path,
            model_key=args.model_key,
            auto_download=False,
            n_threads=args.threads,
            persist_directory=args.db_path,
            retriever_k=args.retriever_k,
            chain_type=args.chain_type
        )

        # Mode interactif ou question unique
        if args.interactive:
            print("Mode interactif activé (tapez 'exit' pour quitter)\n")
            while True:
                try:
                    question = input("\n❓ Question: ").strip()
                    if question.lower() in ['exit', 'quit', 'q']:
                        print("\n👋 Au revoir !")
                        break

                    if not question:
                        continue

                    print("\n⏳ Traitement en cours...")
                    result = rag.query(question)

                    print("\n" + "="*60)
                    print("🤖 RÉPONSE")
                    print("="*60)
                    print(f"\n{result['result']}\n")

                    if args.show_sources and result.get('source_documents'):
                        print("="*60)
                        print("📚 SOURCES")
                        print("="*60)
                        for i, doc in enumerate(result['source_documents'], 1):
                            source = doc.metadata.get('source', 'Unknown')
                            print(f"{i}. {source}")
                        print()

                except KeyboardInterrupt:
                    print("\n\n👋 Au revoir !")
                    break
                except Exception as e:
                    print(f"\n❌ Erreur: {e}\n")
        else:
            # Question unique
            if not args.question:
                print("❌ Aucune question fournie. Utilisez -q ou --interactive")
                return 1

            print(f"❓ Question: {args.question}\n")
            print("⏳ Traitement en cours...")

            result = rag.query(args.question)

            print("\n" + "="*60)
            print("🤖 RÉPONSE")
            print("="*60)
            print(f"\n{result['result']}\n")

            if args.show_sources and result.get('source_documents'):
                print("="*60)
                print("📚 SOURCES")
                print("="*60)
                for i, doc in enumerate(result['source_documents'], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"{i}. {source}")
                print()

        return 0

    except Exception as e:
        print(f"\n❌ Erreur lors de la requête: {e}")
        return 1


def cmd_list_models(args):
    """Liste les modèles disponibles et installés"""
    downloader = LLMDownloader(models_dir=args.models_dir)

    # Afficher les modèles disponibles
    downloader.list_available_models()

    # Afficher les modèles installés
    installed = downloader.list_installed_models()
    if installed:
        print(f"\n✅ Modèles installés localement: {len(installed)}")
        for key, path in installed.items():
            size_mb = Path(path).stat().st_size / (1024**2)
            print(f"   🔹 {key}: {size_mb:.1f} MB")
    else:
        print("\n📭 Aucun modèle installé localement")
        print(f"💡 Téléchargez un modèle avec: python main.py download <model_key>")

    return 0


def cmd_download(args):
    """Télécharge un modèle LLM"""
    print("📥 Gestionnaire de téléchargement LLM\n")

    try:
        downloader = LLMDownloader(models_dir=args.models_dir)

        # Si aucun modèle spécifié, montrer les options
        if not hasattr(args, 'model_key') or not args.model_key:
            downloader.list_available_models()
            print(f"\n💡 Usage: python main.py download <model_key>")
            print(f"Exemple: python main.py download llama-3.2-3b")
            return 1

        # Télécharger le modèle demandé
        success = downloader.download_model(args.model_key, force=getattr(args, 'force', False))

        if success:
            model_path = downloader.get_model_path(args.model_key)
            print(f"\n🎉 Prêt à utiliser!")
            print(f"💡 Testez avec: python main.py --model-path {model_path} query -q 'Test'")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n❌ Erreur lors du téléchargement: {e}")
        return 1


def cmd_clear(args):
    """Efface la base de données vectorielle"""
    print("🗑️  Nettoyage de la base de données\n")

    if not args.force:
        response = input("⚠️  Êtes-vous sûr de vouloir effacer la base ? [y/N]: ")
        if response.lower() not in ['y', 'yes', 'oui', 'o']:
            print("Opération annulée")
            return 0

    try:
        rag = LocalRAG(
            model_path=args.model_path,
            model_key=args.model_key,
            auto_download=False,
            n_threads=args.threads,
            persist_directory=args.db_path,
            retriever_k=args.retriever_k,
            chain_type=args.chain_type
        )
        rag.clear_database()
        print("✅ Base de données effacée avec succès !")
        return 0
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage: {e}")
        return 1


def main():
    """Point d'entrée principal de la CLI"""
    parser = argparse.ArgumentParser(
        description="RAG Local - Système RAG 100% local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Lister les modèles disponibles
  python main.py models

  # Télécharger un modèle
  python main.py download llama-3.2-3b

  # Indexer un document
  python main.py index ./docs/manuel.pdf

  # Indexer un dossier
  python main.py index ./docs/

  # Poser une question
  python main.py query -q "Résume le document"

  # Mode interactif
  python main.py query --interactive

  # Effacer la base de données
  python main.py clear --force
        """
    )

    # Arguments globaux
    parser.add_argument(
        "--model-path",
        help="Chemin vers le modèle GGUF (optionnel)"
    )
    parser.add_argument(
        "--model-key",
        default="llama-3.2-3b",
        help="Clé du modèle recommandé (défaut: llama-3.2-3b)"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        help="Nombre de threads CPU (défaut: 4)"
    )
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Chemin de la base vectorielle (défaut: ./chroma_db)"
    )
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Dossier des modèles (défaut: ./models)"
    )
    parser.add_argument(
        "--retriever-k", "-k",
        type=int,
        default=3,
        help="Nombre de documents à récupérer (défaut: 3, recommandé: 8-12)"
    )
    parser.add_argument(
        "--chain-type",
        default="stuff",
        choices=["stuff", "map_reduce", "refine"],
        help="Stratégie de combinaison des documents (défaut: stuff)"
    )

    # Sous-commandes
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # Commande: init
    parser_init = subparsers.add_parser("init", help="Initialiser le système RAG")
    parser_init.add_argument(
        "--no-download",
        dest="auto_download",
        action="store_false",
        help="Ne pas télécharger le modèle automatiquement"
    )
    parser_init.set_defaults(func=cmd_init, auto_download=True)

    # Commande: index
    parser_index = subparsers.add_parser("index", help="Indexer un document ou dossier")
    parser_index.add_argument("path", help="Chemin du fichier ou dossier à indexer")
    parser_index.set_defaults(func=cmd_index)

    # Commande: query
    parser_query = subparsers.add_parser("query", help="Interroger le système RAG")
    parser_query.add_argument(
        "-q", "--question",
        help="Question à poser"
    )
    parser_query.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Mode interactif"
    )
    parser_query.add_argument(
        "--show-sources",
        action="store_true",
        default=True,
        help="Afficher les sources (défaut: activé)"
    )
    parser_query.set_defaults(func=cmd_query)

    # Commande: models
    parser_models = subparsers.add_parser("models", help="Lister les modèles disponibles")
    parser_models.set_defaults(func=cmd_list_models)

    # Commande: download
    parser_download = subparsers.add_parser("download", help="Télécharger un modèle")
    parser_download.add_argument("model_key", help="Clé du modèle à télécharger")
    parser_download.add_argument("--force", action="store_true", help="Forcer le téléchargement même si déjà présent")
    parser_download.set_defaults(func=cmd_download)

    # Commande: clear
    parser_clear = subparsers.add_parser("clear", help="Effacer la base de données")
    parser_clear.add_argument(
        "--force", "-f",
        action="store_true",
        help="Forcer sans confirmation"
    )
    parser_clear.set_defaults(func=cmd_clear)

    # Parse arguments
    args = parser.parse_args()

    # Si aucune commande, afficher l'aide
    if not args.command:
        parser.print_help()
        return 0

    # Exécuter la commande
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\n👋 Opération annulée par l'utilisateur")
        return 130
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
