# Advanced RAG Engine with Query Decomposition

A production-ready Retrieval Augmented Generation (RAG) system featuring intelligent query decomposition, streaming responses, and enterprise-grade architecture. Built with GitHub Models for seamless AI integration and designed for complex, multi-step reasoning.

## ✨ Key Features

- **🧠 Intelligent Query Decomposition**: Automatically breaks complex questions into focused sub-queries for comprehensive answers
- **⚡ Real-time Streaming**: Perplexity-style streaming responses with live progress indicators
- **🔧 Modular Architecture**: Clean, extensible design with dependency injection and abstract interfaces
- **🚀 Async-First**: Built from the ground up with async/await patterns for optimal performance
- **🎯 Context-Aware**: Multi-turn conversation support with intelligent context management
- **📊 Rich CLI Interface**: Beautiful interactive terminal with progress bars and visual feedback
- **🔍 Advanced Retrieval**: ChromaDB integration with semantic similarity search
- **🤖 GitHub Models**: Access to 40+ state-of-the-art AI models including GPT-4, Llama, and Cohere
- **📈 Enterprise Ready**: Comprehensive error handling, logging, and configuration management

## 🏗️ Architecture

```
src/
├── config/              # Application settings and environment management
├── core/                # Abstract interfaces and base classes
│   ├── interfaces.py    # Core abstractions for modularity
│   └── exceptions.py    # Custom exception hierarchy
├── data/                # Document loading and intelligent processing
│   ├── loader.py        # Multi-format document ingestion
│   └── processor.py     # Smart chunking with overlap handling
├── decomposition/       # Query decomposition engine
│   └── query_decomposer.py  # Smart and context-aware decomposers
├── embeddings/          # HuggingFace embedding implementations
│   └── huggingface_embeddings.py  # Optimized sentence transformers
├── llm/                 # GitHub Models LLM interface
│   └── github_llm.py    # Streaming-first LLM implementation
├── rag/                 # Advanced RAG engine orchestration
│   └── engine.py        # Main engine with decomposition support
└── vectorstore/         # ChromaDB vector storage
    └── chroma_store.py  # Production-ready vector operations
```

## 📋 Prerequisites

- **Python 3.13+**
- **UV package manager** (recommended) or pip
- **GitHub Personal Access Token** with Models access
- **8GB+ RAM** (recommended for optimal performance)

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/cloaky233/rag_new
cd rag_new

# Create and activate virtual environment
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
uv sync
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Required: GitHub Models Configuration
GITHUB_TOKEN=your_github_pat_token_here

# LLM Configuration
LLM_MODEL=meta/Meta-Llama-3.1-405B-Instruct
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# Query Decomposition Settings
ENABLE_QUERY_DECOMPOSITION=true
USE_CONTEXT_AWARE_DECOMPOSER=true
DECOMPOSITION_TEMPERATURE=0.3
MAX_SUB_QUERIES=5

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=rag_collection

# Embedding Configuration
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu

# RAG Configuration
RETRIEVAL_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Logging
LOG_LEVEL=INFO
```

### 3. Prepare Your Documents

```bash
# Create a documents folder
mkdir docs

# Add your documents (supports .txt, .md, .pdf, .docx, .html)
# Example:
echo "Your knowledge base content here..." > docs/sample.txt
```

### 4. Launch Interactive Chat

```bash
# Start the interactive RAG chat
python rag.py chat

# Or with custom document path
python rag.py chat /path/to/your/documents

# View demo questions
python rag.py demo
```

## 🎯 Usage Examples

### Interactive Chat Experience

The system provides a rich, interactive experience similar to Perplexity AI:

```bash
$ python rag.py chat

🤖 Interactive RAG Assistant

Ask complex questions and I'll break them down for comprehensive answers!
Type 'exit' to quit, 'help' for commands

💬 Your Question: What are the benefits and risks of artificial intelligence?

🤔 Analyzing your question...
🔧 Breaking down into 3 focused questions:
   1. What are the main benefits of artificial intelligence?
   2. What are the primary risks and concerns with artificial intelligence?
   3. How do the benefits and risks of AI compare in different domains?

🔍 Researching: What are the main benefits of artificial intelligence?...
🔍 Researching: What are the primary risks and concerns with artificial intelligence?...
🔍 Researching: How do the benefits and risks of AI compare in different domains?...

📚 Research Complete!
🧠 Synthesizing comprehensive answer...

🎯 Final Answer:
[Streaming response appears here in real-time...]

📊 Used 3 research steps • 15 total sources
```

### Programmatic Usage

```python
import asyncio
from src import AdvancedRAGEngine

async def main():
    # Initialize the advanced RAG engine
    rag = AdvancedRAGEngine(use_context_aware_decomposer=True)
    
    # Ingest documents
    doc_count = await rag.ingest_documents("./docs")
    print(f"✅ Ingested {doc_count} document chunks")
    
    # Stream a complex query with decomposition
    question = "Compare machine learning and deep learning approaches"
    
    print("🤖 Assistant: ", end="", flush=True)
    async for chunk in rag.query_with_decomposition_stream(question):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        
        # Access metadata from the first chunk
        if chunk.metadata:
            print(f"\n📊 Decomposed into {chunk.metadata.get('num_sub_queries', 0)} sub-queries")
    
    print()  # New line after completion

if __name__ == "__main__":
    asyncio.run(main())
```

### Simple Streaming Query

```python
async def simple_example():
    rag = AdvancedRAGEngine()
    await rag.ingest_documents("./docs")
    
    # For simple questions, decomposition is automatically skipped
    async for chunk in rag.query_stream("What is machine learning?"):
        print(chunk.content, end="", flush=True)
```

## 🧠 Query Decomposition

The system intelligently determines when to decompose queries based on:

- **Complexity indicators**: "and", "or", "compare", "benefits and risks"
- **Multiple questions**: Queries with multiple question marks
- **Conjunctions**: Multiple "and", "or", "but" statements
- **Length**: Queries longer than 15 words

### Decomposition Examples

| Original Query | Decomposed Sub-queries |
|----------------|----------------------|
| "What are the benefits and risks of solar energy?" | 1. What are the main benefits of solar energy?2. What are the primary risks of solar energy? |
| "Compare supervised and unsupervised learning" | 1. What is supervised learning and how does it work?2. What is unsupervised learning and how does it work?3. What are the key differences between supervised and unsupervised learning? |

## 🔧 Configuration Options

### LLM Models

Choose from 40+ available models:

```env
# High-performance models
LLM_MODEL=meta/Meta-Llama-3.1-405B-Instruct
LLM_MODEL=openai/gpt-4o
LLM_MODEL=cohere/Cohere-command-r-plus

# Efficient models
LLM_MODEL=meta/Meta-Llama-3.1-8B-Instruct
LLM_MODEL=microsoft/Phi-3-medium-4k-instruct
```

### Embedding Models

Optimize for your use case:

```env
# Balanced performance
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# High quality (larger)
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Multilingual support
EMBEDDING_MODEL=BAAI/bge-m3
```

### Advanced Settings

```env
# Query decomposition tuning
DECOMPOSITION_TEMPERATURE=0.3        # Lower = more consistent decomposition
MAX_SUB_QUERIES=5                    # Maximum sub-queries per decomposition
MIN_QUERY_LENGTH_FOR_DECOMPOSITION=15

# Retrieval optimization
RETRIEVAL_K=5                        # Documents per sub-query
CHUNK_SIZE=1000                      # Characters per chunk
CHUNK_OVERLAP=200                    # Overlap between chunks

# Performance tuning
EMBEDDING_DEVICE=cuda                # Use GPU if available
```

## 🛠️ Development

### Adding Custom Components

The modular architecture supports easy extension:

#### Custom LLM Provider

```python
from src.core.interfaces import LLMInterface, StreamingChunk
from typing_extensions import AsyncIterator

class CustomLLM(LLMInterface):
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[StreamingChunk]:
        # Your implementation here
        yield StreamingChunk(content="Hello", is_complete=False)
        yield StreamingChunk(content=" World!", is_complete=True)
    
    async def chat_stream(self, messages: list, **kwargs) -> AsyncIterator[StreamingChunk]:
        # Your chat implementation
        pass

# Use in RAG engine
rag = AdvancedRAGEngine(llm=CustomLLM())
```

#### Custom Query Decomposer

```python
from src.core.interfaces import QueryDecomposerInterface

class CustomDecomposer(QueryDecomposerInterface):
    async def should_decompose(self, query: str) -> bool:
        # Your logic here
        return len(query.split()) > 10
    
    async def decompose_query(self, query: str, context=None) -> list[str]:
        # Your decomposition logic
        return [query]  # Fallback to original

# Use in RAG engine
rag = AdvancedRAGEngine(query_decomposer=CustomDecomposer())
```

### Testing

```bash
# Run interactive tests
python rag.py chat

# Test with demo questions
python rag.py demo

# Custom document path
python rag.py chat /path/to/test/docs --verbose
```

## 📊 Performance Optimization

### Hardware Recommendations

- **CPU**: 4+ cores recommended for concurrent processing
- **RAM**: 8GB+ for large document collections
- **GPU**: Optional, set `EMBEDDING_DEVICE=cuda` if available
- **Storage**: SSD recommended for ChromaDB performance

### Scaling Tips

1. **Batch Size**: Adjust embedding batch size for your hardware
2. **Chunk Strategy**: Optimize chunk size/overlap for your content
3. **Model Selection**: Balance quality vs. speed based on use case
4. **Retrieval K**: Higher K values provide more context but slower processing

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **New LLM Providers**: OpenAI, Anthropic, local models
- **Vector Stores**: Pinecone, Weaviate, Qdrant integration
- **Document Loaders**: Additional format support
- **Query Strategies**: Hybrid search, re-ranking
- **UI Improvements**: Web interface, better visualizations

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Run type checking
mypy src/

# Format code
black src/
isort src/

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GitHub Models** for providing access to state-of-the-art AI models
- **ChromaDB** for efficient vector storage and retrieval
- **HuggingFace** for excellent embedding models and transformers
- **Rich** for beautiful terminal interfaces
- **UV** for lightning-fast dependency management

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/cloaky233/rag_new/issues)
- **Documentation**: [GitHub Models Documentation](https://docs.github.com/en/github-models)
- **Community**: Join our discussions for help and feature requests

---

**Built with ❤️ for intelligent document understanding and complex reasoning**

---

