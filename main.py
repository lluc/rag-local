"""
Exemple d'utilisation du système RAG Local
"""

from rag_local import LocalRAG


def main():
    """Exemple d'initialisation et utilisation du RAG"""
    print("🚀 Initialisation du système RAG Local\n")

    # Initialiser le RAG
    # Le modèle sera téléchargé automatiquement si absent
    rag = LocalRAG(
        # Option 1: Laisser le système gérer automatiquement
        auto_download=True,
        model_key="mistral-7b-instruct-q4",  # Modèle recommandé

        # Option 2: Spécifier un chemin local (si déjà téléchargé)
        # model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",

        n_threads=4  # Ajuster selon votre CPU
    )

    print("\n" + "="*60)
    print("📚 EXEMPLE D'UTILISATION")
    print("="*60)

    # Exemple 1: Indexer un document unique
    print("\n1. Indexer un document:")
    print("   rag.index_document('./docs/manuel.pdf')")

    # Exemple 2: Indexer un dossier complet
    print("\n2. Indexer un dossier:")
    print("   rag.index_directory('./docs')")

    # Exemple 3: Interroger le RAG
    print("\n3. Poser une question:")
    print("   result = rag.query('Résume les points principaux')")
    print("   print(result['result'])")

    # Exemple 4: Voir les sources
    print("\n4. Voir les sources utilisées:")
    print("   for doc in result['source_documents']:")
    print("       print(f\"Source: {doc.metadata['source']}\")")

    print("\n" + "="*60)
    print("✅ Système prêt à l'emploi !")
    print("="*60)

    # Décommentez les lignes suivantes pour un usage réel:

    # # Indexer vos documents
    # rag.index_document("./docs/manuel.pdf")
    # rag.index_document("./docs/notes.md")
    # rag.index_directory("./docs")

    # # Poser des questions
    # result = rag.query("Quelle est la principale idée du document ?")
    # print(f"\n🤖 Réponse: {result['result']}")
    # print(f"\n📚 Basé sur {len(result['source_documents'])} sources")


if __name__ == "__main__":
    main()
