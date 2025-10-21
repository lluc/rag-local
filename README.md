# RAG Local - Système RAG 100% Local

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

Système de Retrieval-Augmented Generation (RAG) complètement local, sans dépendance à des API externes. Utilise des modèles LLM quantifiés (GGUF) et ChromaDB pour la base vectorielle.

## ✨ Fonctionnalités

- 📄 **Traitement de documents** : PDF, Markdown, EPUB
- 🗄️ **Base vectorielle locale** : ChromaDB avec embeddings Sentence-Transformers
- 🤖 **LLM local** : Support des modèles GGUF (llama.cpp)
- 📥 **Téléchargement automatique** : Gestion intelligente des modèles
- 🛡️ **100% privé** : Aucune donnée ne quitte votre machine
- 🖥️ **Interface CLI** : Commandes simples pour indexer et interroger

## 🚀 Installation

### Prérequis

- Python 3.11+
- uv (gestionnaire de paquets)
- curl (pour le téléchargement de modèles)

### Installation rapide

```bash
# Cloner le dépôt
git clone <votre-repo>
cd rag-local

# Installer les dépendances avec uv
uv sync
```

## 📖 Utilisation

### Interface CLI (Recommandée)

```bash
# Lister les modèles disponibles
python main.py models

# Télécharger le modèle recommandé
python main.py download llama-3.2-3b

# Indexer un document
python main.py index ./docs/manuel.pdf

# Indexer un dossier complet
python main.py index ./docs/

# Poser une question
python main.py query -q "Résume le document principal"

# Mode interactif
python main.py query --interactive

# Effacer la base de données
python main.py clear
```

### Utilisation Python

```python
from rag_local import LocalRAG

# Initialiser le RAG (Llama-3.2-3B téléchargé automatiquement)
rag = LocalRAG(
    model_key="llama-3.2-3b",  # Modèle recommandé
    auto_download=True,
    n_threads=4
)

# Indexer des documents
rag.index_document("./docs/manuel.pdf")
rag.index_document("./docs/notes.md")
rag.index_directory("./docs")  # Tout un dossier

# Poser des questions
result = rag.query("Quelle est la principale idée du document ?")
print(f"Réponse: {result['result']}")

# Voir les sources
for doc in result['source_documents']:
    print(f"Source: {doc.metadata['source']}")
```

### Avec un modèle déjà téléchargé

```python
rag = LocalRAG(
    model_path="./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_threads=4
)
```

## 🤖 Gestion des Modèles

### Modèles disponibles

Le système propose 5 modèles pré-configurés :

| Modèle | Taille | Description | Performance |
|--------|--------|-------------|-------------|
| `llama-3.2-3b` | ~1.9GB | **Llama 3.2 3B (RECOMMANDÉ)** - Équilibré | Excellent |
| `mistral-7b-instruct-q4` | ~4.4GB | Mistral 7B - Très performant | Excellent |
| `phi-3-mini` | ~2.4GB | Phi-3 Mini - Compact et rapide | Bon |
| `gemma-2-2b` | ~1.7GB | Gemma 2 2B - Très léger | Bon |
| `tinyllama` | ~700MB | TinyLlama - Ultra-rapide pour tests | Basique |

### CLI de gestion des modèles

```bash
# Voir tous les modèles disponibles et installés
python main.py models

# Télécharger un modèle spécifique
python main.py download mistral-7b-instruct-q4

# Forcer le re-téléchargement
python main.py download llama-3.2-3b --force
```

### Utilisation du LLMDownloader

```python
from llm_downloader import LLMDownloader

downloader = LLMDownloader()

# Afficher les modèles disponibles
downloader.list_available_models()

# Télécharger un modèle spécifique
success = downloader.download_model("llama-3.2-3b")

# Obtenir le chemin d'un modèle installé
model_path = downloader.get_model_path("llama-3.2-3b")
```

## 📁 Structure du Projet

```
rag-local/
├── rag_local.py              # Système RAG principal
├── llm_manager.py            # Gestionnaire avancé de modèles LLM
├── llm_downloader.py         # Module de téléchargement et catalogue
├── main.py                   # Interface CLI complète
├── test_model_selection.py   # Tests de sélection de modèles
├── pyproject.toml            # Dépendances et configuration
├── uv.lock                   # Lock file des dépendances
├── models/                   # Modèles GGUF (créé automatiquement)
├── chroma_db/               # Base vectorielle (créé automatiquement)
└── docs/                    # Vos documents à indexer
```

## ⚙️ Configuration Avancée

### Personnalisation du RAG

```python
rag = LocalRAG(
    model_key="llama-3.2-3b",
    persist_directory="./ma_base_vectorielle",
    embedding_model="all-MiniLM-L6-v2",  # Modèle d'embeddings
    chunk_size=500,                       # Taille des chunks de texte
    chunk_overlap=100,                    # Chevauchement entre chunks
    n_ctx=4096,                          # Taille du contexte LLM
    n_threads=8,                         # Threads CPU
    retriever_k=3,                       # Nombre de documents récupérés
    chain_type="stuff"                   # Stratégie de combinaison
)
```

### Options CLI avancées

```bash
# Configuration globale
python main.py \
  --model-key llama-3.2-3b \
  --threads 8 \
  --db-path ./ma_base \
  --retriever-k 5 \
  query -q "Ma question"

# Afficher les sources dans les réponses
python main.py query -q "Ma question" --show-sources
```

### Formats de documents supportés

- **PDF** : Extraction via PyMuPDF
- **Markdown** : `.md`, `.markdown`
- **EPUB** : Livres électroniques

## 🧪 Tests

```bash
# Tester la sélection de modèles
python test_model_selection.py

# Vérifier l'import des modules
python -c "from rag_local import LocalRAG; print('✅ Import réussi')"

# Tester l'affichage des modèles disponibles
python -c "from llm_downloader import LLMDownloader; LLMDownloader().list_available_models()"
```

## 🔧 Dépannage

### Erreur de mémoire lors du chargement du modèle

Essayez un modèle plus petit :
```bash
python main.py download gemma-2-2b  # ~1.7GB
python main.py download tinyllama   # ~700MB
```

### Le téléchargement échoue

Téléchargez manuellement depuis Hugging Face :

```bash
mkdir -p models
cd models
curl -L -O "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

### Erreurs d'imports LangChain

Le projet utilise LangChain 1.0+ avec les nouveaux packages modulaires :
- `langchain-core`
- `langchain-text-splitters`
- `langchain-community`
- `langchain-classic`
- `langchain-chroma`
- `langchain-huggingface`

### Réponses de mauvaise qualité en français

Le système utilise désormais Llama-3.2-3B par défaut, optimisé pour les réponses en français. Si vous utilisez encore Mistral-7B, passez à Llama-3.2-3B :

```bash
python main.py download llama-3.2-3b
python main.py query -q "Test en français"
```

## 📊 Performance

- **Embeddings** : ~100 documents/seconde (CPU)
- **LLM Inference** : 4-20 tokens/s selon le modèle et CPU
- **Base vectorielle** : ChromaDB optimisé pour recherche locale
- **Modèle recommandé** : Llama-3.2-3B (excellent compromis taille/qualité)

## 🔒 Confidentialité

Ce système est **100% local** :
- ✅ Aucune donnée envoyée à des API externes
- ✅ Tous les calculs sur votre machine
- ✅ Modèles stockés localement
- ✅ Base vectorielle locale
- ✅ Interface CLI sans connexion réseau

## 📚 Ressources

- [LangChain Documentation](https://python.langchain.com/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [ChromaDB](https://www.trychroma.com/)
- [Modèles GGUF](https://huggingface.co/models?search=gguf)
- [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)

## 📝 Exemples d'Usage

### Analyse de documentation technique

```bash
# Indexer votre documentation
python main.py index ./docs/technique/

# Poser des questions spécifiques
python main.py query -q "Comment configurer l'authentification ?"
python main.py query -q "Quelles sont les API disponibles ?"
```

### Recherche dans des livres

```bash
# Indexer des EPUB
python main.py index ./bibliotheque/*.epub

# Mode interactif pour exploration
python main.py query --interactive
```

### Analyse de code source

```bash
# Indexer des fichiers Markdown de documentation
python main.py index ./src/docs/

# Rechercher des patterns
python main.py query -q "Comment utiliser la fonction de cache ?"
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une PR.

## 📄 Licence

MIT License

Copyright (c) 2024 RAG Local

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.