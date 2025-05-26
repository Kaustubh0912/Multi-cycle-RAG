# RAG Engine with GitHub Models

A production-ready Retrieval Augmented Generation (RAG) system built with GitHub Models, featuring modular architecture, async support, and enterprise-grade capabilities.



- **GitHub Models Integration**: Seamless integration with 40+ AI models including OpenAI, Meta, and DeepSeek
- **Modular Architecture**: Clean, extensible design with abstract interfaces
- **Async Support**: Built from the ground up with async/await patterns
- **Smart Document Processing**: Intelligent chunking with recursive text splitting
- **Vector Search**: ChromaDB integration with hybrid search capabilities
- **HuggingFace Embeddings**: Optimized sentence transformers with torch compilation
- **Enterprise Ready**: Comprehensive error handling, logging, and configuration management
- **Zero Infrastructure**: No complex setup required - just your GitHub token

## 🏗️ Architecture

```
src/
├── config/          # Application settings and environment management
├── core/            # Abstract interfaces and base classes
├── data/            # Document loading and processing
├── embeddings/      # HuggingFace embedding implementations
├── llm/             # GitHub Models LLM interface
├── rag/             # Main RAG engine orchestration
└── vectorstore/     # ChromaDB vector storage
```

## 📋 Prerequisites

- Python 3.13+
- UV package manager
- GitHub Personal Access Token with Models access

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/cloaky233/rag_new.git
cd rag_new
```

### 2. Setup Environment

```bash
# Create virtual environment
uv venv

# Activate environment (Windows)
.venv\Scripts\activate
# Or on macOS/Linux
source .venv/bin/activate

# Install dependencies
uv sync
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
GITHUB_TOKEN=your_github_pat_token_here
LLM_MODEL=meta/Meta-Llama-3.1-405B-Instruct
LLM_TEMPERATURE=0.4
LLM_MAX_TOKENS=10000
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=rag_collection
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
RETRIEVAL_K=15
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
```

### 4. Run the Application

```python
import asyncio
from src import RAGEngine

async def main():
    # Initialize RAG engine
    rag = RAGEngine()

    # Ingest documents
    doc_count = await rag.ingest_documents("./your_documents_folder")
    print(f"Ingested {doc_count} documents")

    # Query the system
    result = await rag.query("What is the main topic of the documents?")
    print(f"Answer: {result.response}")
    print(f"Sources: {len(result.source_documents)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔧 Configuration

The application uses Pydantic settings for configuration management. All settings can be configured via environment variables:

| Setting | Description | Default |
|---------|-------------|---------|
| `GITHUB_TOKEN` | GitHub Personal Access Token | Required |
| `LLM_MODEL` | GitHub Models model identifier | `cohere/Cohere-command-r` |
| `LLM_TEMPERATURE` | Model temperature for generation | `0.7` |
| `LLM_MAX_TOKENS` | Maximum tokens per response | `1000` |
| `RETRIEVAL_K` | Number of documents to retrieve | `5` |
| `CHUNK_SIZE` | Document chunk size in characters | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `BAAI/bge-small-en-v1.5` |

## 🔍 Usage Examples

### Basic RAG Query

```python
from src import RAGEngine

async def example_query():
    rag = RAGEngine()

    # Ingest documents from a directory
    await rag.ingest_documents("./docs")

    # Ask questions about your documents
    result = await rag.query("How do I configure the system?")

    print(f"Answer: {result.response}")
    for i, doc in enumerate(result.source_documents):
        print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}")
```

### Custom LLM Configuration

```python
from src.llm import GitHubLLM
from src import RAGEngine

async def custom_llm_example():
    # Create custom LLM with specific settings
    llm = GitHubLLM()

    # Initialize RAG with custom LLM
    rag = RAGEngine(llm=llm)

    # Use with custom parameters
    result = await rag.query(
        "Explain the architecture",
        k=10  # Retrieve more documents
    )
```

### Document Processing

```python
from src.data import DocumentLoader, DocumentProcessor

async def process_documents():
    loader = DocumentLoader()
    processor = DocumentProcessor()

    # Load documents
    documents = await loader.load_from_directory("./my_docs")

    # Process with chunking
    processed = await processor.process_documents(documents)

    print(f"Original: {len(documents)} docs")
    print(f"Processed: {len(processed)} chunks")
```

## 🧪 Available Models

The system supports 40+ models from GitHub Models including:

- **OpenAI**: GPT-4o, GPT-4o-mini
- **Meta**: Llama 3.1 (8B, 70B, 405B)
- **Microsoft**: Phi-3 series
- **Cohere**: Command R, Command R+
- **DeepSeek**: DeepSeek-V2

## 🛠️ Development

### Project Structure

```
rag_new/
├── .venv/              # Virtual environment
├── chroma_db/          # Vector database storage
├── docs/               # Documentation
├── src/                # Source code
│   ├── config/         # Configuration management
│   ├── core/           # Core interfaces and exceptions
│   ├── data/           # Document loading and processing
│   ├── embeddings/     # Embedding implementations
│   ├── llm/            # Language model interfaces
│   ├── rag/            # Main RAG engine
│   └── vectorstore/    # Vector storage implementations
├── .env                # Environment variables
├── .gitignore          # Git ignore rules
├── .python-version     # Python version specification
├── pyproject.toml      # Project configuration
├── README.md           # This file
├── test_rag.py         # Test script
└── uv.lock             # Dependency lock file
```

### Adding New Components

The modular architecture makes it easy to extend:

1. **New LLM Provider**: Implement `LLMInterface`
2. **New Vector Store**: Implement `VectorStoreInterface`
3. **New Embeddings**: Implement `EmbeddingInterface`

### Testing

```bash
# Run the test script
python test_rag.py
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [GitHub Models](https://github.com/features/models) for seamless AI integration
- Powered by [Azure AI Inference](https://azure.microsoft.com/en-us/products/ai-services/) for enterprise-grade reliability
- Uses [ChromaDB](https://www.trychroma.com/) for efficient vector storage
- Dependency management by [UV](https://github.com/astral-sh/uv) for lightning-fast operations

## 📞 Support

- **Creator**: Lay Sheth ([@cloaky233](https://github.com/cloaky233))
- **Issues**: [GitHub Issues](https://github.com/cloaky233/rag_new/issues)
- **Documentation**: [GitHub Models Documentation](https://docs.github.com/en/github-models)

---

**Built with ❤️ using GitHub Models**
