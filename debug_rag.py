#!/usr/bin/env python3
"""
Debug complet du système RAG pour identifier les problèmes
"""

import sys
from pathlib import Path
from rag_local import LocalRAG

def debug_retrieval(question: str):
    """Debug complet du système de retrieval"""
    print("🔬 DEBUG COMPLET DU RAG")
    print("=" * 80)

    try:
        # Initialiser le RAG
        rag = LocalRAG(
            model_key="llama-3.2-3b",
            auto_download=False,
            n_threads=4,
            persist_directory="./chroma_db",
            retriever_k=8
        )

        # 1. Statistiques de base
        stats = rag.get_database_stats()
        print(f"📊 Base de données:")
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Chunks: {stats['total_chunks']}")
        for doc in stats['documents']:
            print(f"   📄 {Path(doc).name}")

        print(f"\n🔍 Question analysée: '{question}'")

        # 2. Test de recherche vectorielle directe
        print(f"\n🎯 RECHERCHE VECTORIELLE DIRECTE:")
        print("-" * 50)

        docs = rag.vectorstore.similarity_search(question, k=8)
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            doc_type = doc.metadata.get('document_type', 'N/A')
            content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content

            print(f"\n{i}. Source: {Path(source).name}")
            print(f"   Type: {doc_type}")
            print(f"   Contenu: {content}")

        # 3. Test de recherche de métadonnées
        print(f"\n🏷️  RECHERCHE MÉTADONNÉES:")
        print("-" * 50)

        metadata_docs = rag.vectorstore.similarity_search("DOCUMENT TYPE DESCRIPTION", k=5)
        metadata_found = [doc for doc in metadata_docs if "TYPE:" in doc.page_content or "DOCUMENT:" in doc.page_content]

        if metadata_found:
            for i, doc in enumerate(metadata_found, 1):
                print(f"\n{i}. Métadonnées trouvées:")
                print(f"   {doc.page_content}")
        else:
            print("❌ Aucun chunk de métadonnées trouvé!")
            print("💡 Les documents doivent être re-indexés avec la nouvelle version")

        # 4. Test du bypass
        print(f"\n🔄 TEST DU BYPASS:")
        print("-" * 50)

        nature_questions = ["de quoi parle", "quel type", "qu'est-ce que", "que contient", "nature du document"]
        bypass_triggered = any(phrase in question.lower() for phrase in nature_questions)
        print(f"Bypass activé: {bypass_triggered}")

        if bypass_triggered and metadata_found:
            for doc in metadata_found:
                if "TYPE: Code du travail français" in doc.page_content:
                    print("✅ Détection: Code du travail français")
                    print("✅ Réponse attendue: 'Le document LEGITEXT est le Code du travail français.'")
                    break
            else:
                print("❌ Aucune détection de type dans les métadonnées")

        # 5. Test de la requête complète
        print(f"\n🤖 REQUÊTE COMPLÈTE:")
        print("-" * 50)

        result = rag.query(question)
        print(f"Réponse: {result['result']}")

        print(f"\n📚 Sources de la réponse:")
        for i, doc in enumerate(result.get('source_documents', []), 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"   {i}. {Path(source).name}")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    question = "De quoi parle le document LEGITEXT ?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    debug_retrieval(question)