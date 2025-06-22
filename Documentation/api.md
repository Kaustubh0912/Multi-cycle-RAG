## API Documentation (`api.md`)

This API documentation describes the main interfaces, classes, and methods of the Reflexion RAG Engine. It is designed for developers integrating, extending, or operating the system.

---

### Table of Contents

- [Overview](#overview)
- [Core Classes](#core-classes)
  - [RAGEngine](#ragengine)
  - [Document](#document)
  - [StreamingChunk](#streamingchunk)
  - [ReflexionMemory](#reflexionmemory)
- [Key Interfaces](#key-interfaces)
  - [LLMInterface](#llminterface)
  - [VectorStoreInterface](#vectorstoreinterface)
  - [WebSearchInterface](#websearchinterface)
  - [EmbeddingInterface](#embeddinginterface)
- [Document Management](#document-management)
  - [DocumentLoader](#documentloader)
  - [DocumentProcessor](#documentprocessor)
- [Vector Store](#vector-store)
- [Web Search](#web-search)
- [Memory Cache](#memory-cache)
- [Exceptions](#exceptions)

---

## Overview

The Reflexion RAG Engine is a modular, extensible retrieval-augmented generation system with self-reflexion, multi-cycle reasoning, and hybrid (vector + web) retrieval. It exposes a Python API with async support for ingestion, querying, and administration tasks[1].

---

## Core Classes

### RAGEngine

Main entry point for most users. Wraps the full reflexion architecture and provides a simplified interface.

```python
from rag.src.rag.engine import RAGEngine
```

#### Constructor

```python
RAGEngine(
    generation_llm: Optional[LLMInterface] = None,
    vector_store: Optional[VectorStoreInterface] = None,
    **kwargs
)
```
- `generation_llm`: Optional custom LLM for answer generation.
- `vector_store`: Optional custom vector store.
- `**kwargs`: Passed to internal engine.

#### Methods

- **`async ingest_documents(directory_path: str) -> int`**
  Ingests all supported documents from a directory. Returns the number of successfully ingested documents.

- **`async query_stream(question: str, k: Optional[int] = None) -> AsyncIterator[StreamingChunk]`**
  Processes a query using the reflexion loop, yielding streaming response chunks.
  - `question`: The user query.
  - `k`: (Optional) Override for number of documents to retrieve.

- **`get_engine_info() -> Dict`**
  Returns engine configuration and memory statistics.

- **`async clear_memory_cache() -> None`**
  Clears the reflexion memory cache.

- **`async count_documents() -> int`**
  Returns the number of documents in the vector store.

- **`async delete_all_documents(confirm_string: str = "CONFIRM") -> bool`**
  Deletes all documents from the vector store (requires confirmation).

---

### Document

Standard document format used throughout the system.

```python
from rag.src.core.interfaces import Document

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
```

---

### StreamingChunk

Represents a chunk of a streaming response.

```python
@dataclass
class StreamingChunk:
    content: str
    is_complete: bool = False
    usage_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
```

---

### ReflexionMemory

Stores the full history and results of a reflexion loop for a query.

```python
@dataclass
class ReflexionMemory:
    original_query: str
    cycles: List[ReflexionCycle] = field(default_factory=list)
    final_answer: str = ""
    total_processing_time: float = 0.0
    total_documents_retrieved: int = 0
    total_web_results_retrieved: int = 0
```

---

## Key Interfaces

### LLMInterface

Abstract interface for all LLM implementations.

```python
class LLMInterface(ABC):
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[StreamingChunk]:
        pass

    @abstractmethod
    def chat_stream(self, messages: list[Dict[str, str]], **kwargs: Any) -> AsyncIterator[StreamingChunk]:
        pass
```

---

### VectorStoreInterface

Abstract interface for all vector store backends.

```python
class VectorStoreInterface(ABC):
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        pass

    @abstractmethod
    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        pass

    @abstractmethod
    async def count_documents(self) -> int:
        pass

    @abstractmethod
    async def delete_all_documents(self, confirm_string: str) -> bool:
        pass

    @abstractmethod
    async def add_web_search_results(self, web_results: List[WebSearchResult]) -> List[str]:
        pass

    @abstractmethod
    async def similarity_search_combined(self, query: str, k_docs: int = 3, k_web: int = 2) -> List[Document]:
        pass
```

---

### WebSearchInterface

Abstract interface for web search providers.

```python
class WebSearchInterface(ABC):
    @abstractmethod
    async def search_and_extract(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        pass
```

---

### EmbeddingInterface

Abstract interface for embedding models.

```python
class EmbeddingInterface(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
```

---

## Document Management

### DocumentLoader

Loads and parses documents from disk.

```python
from rag.src.data.loader import DocumentLoader
```

- **`async load_from_directory(directory_path: str) -> List[Document]`**
  Loads and parses all supported files from a directory.

- **`async load_from_text(text: str, metadata: Optional[Dict[str, Any]] = None) -> Document`**
  Wraps a text string as a Document.

---

### DocumentProcessor

Splits and sanitizes documents for embedding.

```python
from rag.src.data.processor import DocumentProcessor
```

- **`async process_documents(documents: List[Document]) -> List[Document]`**
  Splits documents into chunks using intelligent, recursive splitting with overlap and metadata propagation.

---

## Vector Store

The default implementation is `SurrealDBVectorStore`, which supports:

- Adding documents and web search results with embeddings.
- Vector similarity search (documents, web, or combined).
- Document counting and deletion.
- Schema management and connection pooling.

---

## Web Search

The default implementation is `GoogleWebSearch`, which supports:

- Google Custom Search API queries.
- Content extraction from result URLs using multiple strategies.
- Quality filtering and fallback to snippets if extraction fails.

---

## Memory Cache

The `ReflexionMemoryCache` provides LRU caching for reflexion loop results.

- `get(query_hash: str) -> Optional[ReflexionMemory]`
- `put(query_hash: str, memory: ReflexionMemory) -> None`
- `clear() -> None`
- `get_stats() -> Dict[str, Any]`

---

## Exceptions

All exceptions inherit from `RAGException`:

- `LLMException`
- `VectorStoreException`
- `EmbeddingException`
- `DocumentProcessingException`
- `WebSearchException`
- `WebSearchConfigurationException`
- `WebSearchAPIException`
- `ContentExtractionException`

---

## Example Usage

```python
from rag.src.rag.engine import RAGEngine

engine = RAGEngine()

# Ingest documents
await engine.ingest_documents("./docs")

# Query with reflexion (streaming)
async for chunk in engine.query_stream("What is retrieval-augmented generation?"):
    print(chunk.content, end="")
    if chunk.is_complete:
        print("\n---\nFinal answer delivered.")
```
