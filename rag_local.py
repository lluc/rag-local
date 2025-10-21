"""
RAG Local - Système de Retrieval-Augmented Generation
Support: PDF, Markdown, EPUB
"""

import os
from pathlib import Path
from typing import List, Optional
import pymupdf
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

from llm_manager import LLMManager


class DocumentLoader:
    """Charge et extrait le texte des documents PDF, Markdown et EPUB"""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extrait le texte d'un PDF"""
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    @staticmethod
    def load_markdown(file_path: str) -> str:
        """Charge un fichier Markdown"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_epub(file_path: str) -> str:
        """Extrait le texte d'un EPUB"""
        book = epub.read_epub(file_path)
        text = ""
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + "\n\n"
        
        return text
    
    @staticmethod
    def load_document(file_path: str) -> str:
        """Détecte le type et charge le document approprié"""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == '.pdf':
            return DocumentLoader.load_pdf(file_path)
        elif ext in ['.md', '.markdown']:
            return DocumentLoader.load_markdown(file_path)
        elif ext == '.epub':
            return DocumentLoader.load_epub(file_path)
        else:
            raise ValueError(f"Format non supporté: {ext}")


class LocalRAG:
    """Système RAG local avec ChromaDB et llama-cpp-python"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_key: str = "llama-3.2-3b",
        auto_download: bool = True,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        n_ctx: int = 4096,
        n_threads: int = 8,
        language: str = "fr",
        retriever_k: int = 3,
        chain_type: str = "stuff",
    ):
        """
        Initialise le système RAG

        Args:
            model_path: Chemin vers le modèle GGUF (optionnel)
            model_key: Clé du modèle recommandé à télécharger si model_path absent
            auto_download: Télécharger automatiquement le modèle si absent
            persist_directory: Dossier de persistance ChromaDB
            embedding_model: Modèle sentence-transformers
            chunk_size: Taille des chunks de texte
            chunk_overlap: Chevauchement entre chunks
            n_ctx: Taille du contexte pour le LLM
            n_threads: Nombre de threads CPU
        """
        self.persist_directory = persist_directory
        self.language = language
        self.retriever_k = retriever_k
        self.chain_type = chain_type

        # Cache de traduction pour accélérer les traductions répétées
        self._translation_cache = {}

        # Configuration des embeddings
        print("Chargement du modèle d'embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # Configuration du text splitter avec meilleure capture des métadonnées
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True  # Garder les séparateurs pour le contexte
        )

        # Initialisation ChromaDB
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

        # Gestion du modèle LLM avec LLMManager
        llm_manager = LLMManager()
        resolved_model_path = llm_manager.get_or_download_model(
            model_path=model_path,
            model_key=model_key,
            auto_download=auto_download
        )

        # Configuration du LLM équilibré (éviter hallucinations tout en étant informatif)
        print(f"Chargement du modèle LLM: {resolved_model_path}")
        self.llm = LlamaCpp(
            model_path=resolved_model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            temperature=0.01,        # Maximum de contrôle
            max_tokens=200,          # Augmenté pour éviter troncature
            n_batch=512,             # Traitement par batch efficace
            repeat_penalty=3.0,      # Pénalité extrême contre répétitions
            top_p=0.3,              # Sampling très restrictif
            top_k=15,               # Choix très limité
            stop=["\n\nQuestion:", "EXTRAITS:", "4.", "\n4", "La réponse", "donc :", "final"],
            verbose=False,
        )

        # Configuration LLM strict pour les questions de métadonnées
        self.llm_strict = LlamaCpp(
            model_path=resolved_model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            temperature=0.0,         # Température zéro = réponses déterministes
            max_tokens=30,           # Très peu de tokens pour forcer la concision
            n_batch=512,
            repeat_penalty=2.0,
            top_p=0.5,
            top_k=10,
            stop=["\n", ".", "Question:"],
            verbose=False,
        )

        # Chains RAG
        self.qa_chain = None
        self.qa_chain_strict = None
        self._setup_chain()
    
    def _setup_chain(self):
        """Configure la chaîne RetrievalQA avec prompt français optimisé"""
        if self.vectorstore._collection.count() > 0:
            # Stratégie de recherche hybride pour capturer différents types d'informations
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance pour plus de diversité
                search_kwargs={
                    "k": getattr(self, 'retriever_k', 8),
                    "lambda_mult": 0.5  # Balance pertinence/diversité
                }
            )

            # Prompt simple pour questions sur le contenu
            french_prompt = """Utilise les extraits pour répondre à la question en français avec 3 points DIFFÉRENTS.

Extraits:
{context}

Question: {question}

Réponse (3 droits distincts):
1."""

            from langchain_core.prompts import PromptTemplate
            PROMPT = PromptTemplate(
                template=french_prompt,
                input_variables=["context", "question"]
            )

            # Chain principale avec LLM équilibré
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            # Chain stricte pour les questions de métadonnées
            strict_prompt = """Réponds UNIQUEMENT avec ce qui est dans les extraits.

RÈGLES:
- Si tu vois "TYPE: Code du travail français" → dis "Le document est le Code du travail français"
- Si tu vois "TYPE: Manuel Bash" → dis "Le document est un manuel Bash"
- JAMAIS d'invention ou d'interprétation
- Maximum 1 phrase

EXTRAITS:
{context}

QUESTION: {question}

RÉPONSE:"""

            from langchain_core.prompts import PromptTemplate
            STRICT_PROMPT = PromptTemplate(
                template=strict_prompt,
                input_variables=["context", "question"]
            )

            self.qa_chain_strict = RetrievalQA.from_chain_type(
                llm=self.llm_strict,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": STRICT_PROMPT}
            )
    
    def _detect_document_type(self, text: str, file_path: str) -> str:
        """Détecte le type de document basé sur le contenu et le nom de fichier"""
        text_lower = text.lower()
        filename = Path(file_path).name.lower()

        # Détection pour les codes juridiques
        if "legitext" in filename or "code du travail" in text_lower or "code de travail" in text_lower:
            return "Code du travail français"
        elif "code civil" in text_lower:
            return "Code civil français"
        elif "code pénal" in text_lower:
            return "Code pénal français"
        elif "bash" in filename or "shell" in text_lower:
            return "Manuel Bash/Shell"
        elif ".pdf" in filename:
            return "Document PDF"
        else:
            return "Document général"

    def index_document(self, file_path: str, metadata: Optional[dict] = None):
        """
        Indexe un document dans la base vectorielle

        Args:
            file_path: Chemin vers le document
            metadata: Métadonnées optionnelles
        """
        print(f"Indexation de: {file_path}")

        # Charger le document
        text = DocumentLoader.load_document(file_path)

        # Détecter le type de document
        doc_type = self._detect_document_type(text, file_path)
        print(f"📄 Type détecté: {doc_type}")

        # Créer les chunks
        chunks = self.text_splitter.split_text(text)
        
        # Préparer les métadonnées enrichies
        base_metadata = {
            "source": file_path,
            "document_type": doc_type,
            "filename": Path(file_path).name
        }
        if metadata:
            base_metadata.update(metadata)
        
        # Créer un chunk spécial de métadonnées pour identifier le document
        metadata_chunk = f"""DOCUMENT: {Path(file_path).name}
TYPE: {doc_type}
DESCRIPTION: Ce document est un {doc_type} contenant des informations officielles.
SOURCE: {file_path}
NOMBRE_DE_PAGES: {len(chunks)} sections indexées"""

        # Créer les documents avec le chunk de métadonnées en premier
        documents = [Document(page_content=metadata_chunk, metadata=base_metadata)]

        # Ajouter les chunks de contenu
        documents.extend([
            Document(page_content=chunk, metadata=base_metadata)
            for chunk in chunks
        ])

        # Ajouter à la base vectorielle par batches pour éviter les erreurs de taille
        batch_size = 500  # Taille de batch plus petite pour les gros documents
        total_docs = len(documents)

        print(f"📊 {total_docs} chunks créés, traitement par batches de {batch_size}")

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size

            print(f"   ⏳ Batch {batch_num}/{total_batches} : {len(batch)} documents")

            try:
                self.vectorstore.add_documents(batch)
                print(f"   ✅ Batch {batch_num} ajouté avec succès")
            except Exception as e:
                print(f"   ❌ Erreur batch {batch_num}: {e}")
                # Tenter avec un batch plus petit
                if len(batch) > 100:
                    smaller_batch_size = 100
                    print(f"   🔄 Réessai avec batch de {smaller_batch_size}")
                    for j in range(0, len(batch), smaller_batch_size):
                        small_batch = batch[j:j + smaller_batch_size]
                        self.vectorstore.add_documents(small_batch)
                        print(f"      ✅ Sous-batch {j//smaller_batch_size + 1} traité")
                else:
                    raise e
        
        # Reconfigurer la chain
        self._setup_chain()

        print(f"\n✅ Indexation terminée : {total_docs} chunks traités avec succès")
    
    def index_directory(self, directory: str):
        """Indexe tous les documents supportés dans un dossier"""
        path = Path(directory)
        supported_extensions = ['.pdf', '.md', '.markdown', '.epub']
        
        for file_path in path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    self.index_document(str(file_path))
                except Exception as e:
                    print(f"✗ Erreur avec {file_path}: {e}")

    def _get_metadata_chunks(self, question: str) -> List[Document]:
        """Récupère spécifiquement les chunks de métadonnées pour identifier les documents"""
        try:
            # Recherche directe dans la collection pour les chunks de métadonnées
            collection = self.vectorstore._collection
            all_docs = collection.get(include=['documents', 'metadatas'])

            metadata_docs = []
            for doc_content, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                if "TYPE:" in doc_content and "DOCUMENT:" in doc_content:
                    # Reconstruct Document object
                    from langchain_core.documents import Document
                    metadata_docs.append(Document(page_content=doc_content, metadata=metadata))

            return metadata_docs[:2]  # Max 2 chunks de métadonnées
        except Exception as e:
            print(f"Erreur recherche métadonnées: {e}")
            return []

    def query(self, question: str) -> dict:
        """
        Pose une question au RAG avec réponse directe en français

        Args:
            question: Question en langage naturel

        Returns:
            dict avec 'result' et 'source_documents'
            Réponse optimisée pour éviter les redondances
        """
        if not self.qa_chain:
            return {
                "result": "Aucun document n'est indexé. Utilisez index_document() d'abord.",
                "source_documents": []
            }

        # Détecter les questions sur la NATURE du document (pas le contenu)
        document_nature_patterns = [
            "de quoi parle le document",
            "de quoi parle ce document",
            "quel type de document",
            "qu'est-ce que le document",
            "qu'est-ce que ce document",
            "que contient le document",
            "nature du document",
            "type de document"
        ]
        is_nature_question = any(pattern in question.lower() for pattern in document_nature_patterns)

        if is_nature_question:
            # 1. Essayer d'abord le bypass direct avec métadonnées
            metadata_docs = self._get_metadata_chunks(question)
            if metadata_docs:
                for doc in metadata_docs:
                    if "TYPE: Code du travail français" in doc.page_content:
                        return {
                            "result": "Le document LEGITEXT est le Code du travail français.",
                            "source_documents": metadata_docs
                        }
                    elif "TYPE: Manuel Bash" in doc.page_content:
                        return {
                            "result": "Le document est un manuel Bash/Shell.",
                            "source_documents": metadata_docs
                        }

            # 2. Si pas de métadonnées, utiliser la chaîne stricte
            if hasattr(self, 'qa_chain_strict') and self.qa_chain_strict:
                return self.qa_chain_strict.invoke({"query": question})

        # 3. Pour toutes les autres questions, utiliser la chaîne normale (assouplie)
        return self.qa_chain.invoke({"query": question})
    
    def list_indexed_documents(self) -> List[str]:
        """
        Liste les documents indexés dans la base vectorielle

        Returns:
            Liste des chemins de fichiers uniques
        """
        try:
            # Récupérer tous les documents
            collection = self.vectorstore._collection
            all_metadata = collection.get()["metadatas"]

            # Extraire les sources uniques
            sources = set()
            for metadata in all_metadata:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])

            return sorted(list(sources))
        except Exception:
            return []

    def get_database_stats(self) -> dict:
        """
        Retourne les statistiques de la base de données

        Returns:
            Dictionnaire avec le nombre de documents et chunks
        """
        try:
            collection = self.vectorstore._collection
            total_chunks = collection.count()
            sources = self.list_indexed_documents()

            return {
                "total_documents": len(sources),
                "total_chunks": total_chunks,
                "documents": sources
            }
        except Exception:
            return {"total_documents": 0, "total_chunks": 0, "documents": []}

    def clear_database(self):
        """Supprime toutes les données de la base vectorielle"""
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.qa_chain = None
        print("✓ Base de données effacée")


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # À adapter
    
    # Vérifier que le modèle existe
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ Modèle non trouvé: {MODEL_PATH}")
        print("Télécharge un modèle GGUF depuis Hugging Face")
        print("Exemple: https://huggingface.co/TheBloke")
        exit(1)
    
    # Initialiser le RAG
    rag = LocalRAG(
        model_path=MODEL_PATH,
        persist_directory="./chroma_db",
        n_threads=4,  # Ajuste selon ton CPU
    )
    
    # Indexer des documents
    # rag.index_document("./docs/mon_document.pdf")
    # rag.index_directory("./docs")
    
    # Poser des questions
    # result = rag.query("Quelle est la principale idée du document ?")
    # print(f"\n🤖 Réponse: {result['result']}")
    # print(f"\n📚 Sources: {len(result['source_documents'])} documents")
