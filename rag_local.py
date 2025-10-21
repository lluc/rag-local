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

        # Configuration du text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
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

        # Configuration du LLM
        print(f"Chargement du modèle LLM: {resolved_model_path}")
        self.llm = LlamaCpp(
            model_path=resolved_model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            temperature=0.3,      # Plus déterministe
            max_tokens=400,       # Limiter la longueur pour éviter les dérives
            n_batch=512,          # Traitement par batch efficace
            repeat_penalty=1.3,   # Pénaliser fortement les répétitions
            top_p=0.9,            # Nucleus sampling pour plus de diversité
            top_k=40,             # Limiter les tokens candidats
            verbose=False,
        )

        # Chain RAG
        self.qa_chain = None
        self._setup_chain()
    
    def _setup_chain(self):
        """Configure la chaîne RetrievalQA avec prompt français optimisé"""
        if self.vectorstore._collection.count() > 0:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": getattr(self, 'retriever_k', 3)}
            )

            # Prompt français optimisé pour éviter les redondances
            french_prompt = """Tu es un assistant qui répond en français de manière CONCISE et DIRECTE.

INSTRUCTIONS STRICTES:
1. Réponds à la question en UNE SEULE FOIS
2. NE RÉPÈTE JAMAIS la même information
3. Structure: introduction brève + points clés + conclusion courte
4. Maximum 200 mots
5. ARRÊTE-TOI dès que la réponse est complète
6. Si tu ne sais pas, dis "Information non disponible" et STOP

Documents de référence:
{context}

Question: {question}

Réponse directe (max 200 mots):"""

            from langchain_core.prompts import PromptTemplate
            PROMPT = PromptTemplate(
                template=french_prompt,
                input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
    
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
        
        # Créer les chunks
        chunks = self.text_splitter.split_text(text)
        
        # Préparer les métadonnées
        base_metadata = {"source": file_path}
        if metadata:
            base_metadata.update(metadata)
        
        # Créer les documents
        documents = [
            Document(page_content=chunk, metadata=base_metadata)
            for chunk in chunks
        ]
        
        # Ajouter à la base vectorielle
        self.vectorstore.add_documents(documents)
        
        # Reconfigurer la chain
        self._setup_chain()
        
        print(f"✓ {len(chunks)} chunks indexés")
    
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

        return self.qa_chain.invoke({"query": question})
    
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
