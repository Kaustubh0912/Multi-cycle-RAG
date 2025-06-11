# API Documentation

This document provides comprehensive documentation for the Reflexion RAG Engine API, including both Python SDK usage and future REST API endpoints.

## Python SDK API

### Core Classes

#### RAGEngine

The main interface for interacting with the Reflexion RAG system.

```python
from rag.src.rag.engine import RAGEngine
import asyncio

# Initialize the engine
engine = RAGEngine()
```

##### Methods

###### `ingest_documents(directory_path: str) -> int`

Ingest documents from a specified directory.

**Parameters:**
- `directory_path` (str): Path to directory containing documents

**Returns:**
- `int`: Number of documents successfully ingested

**Example:**
```python
async def ingest_docs():
    engine = RAGEngine()
    count = await engine.ingest_documents("./docs")
    print(f"Ingested {count} documents")

asyncio.run(ingest_docs())
```

**Supported Formats:**
- PDF (`.pdf`)
- Text files (`.txt`)
- Microsoft Word (`.docx`)
- Markdown (`.md`)
- HTML (`.html`)
- Rich Text Format (`.rtf`)

---

###### `query_stream(question: str, k: Optional[int] = None) -> AsyncIterator[StreamingChunk]`

Process a query using reflexion architecture with streaming response.

**Parameters:**
- `question` (str): User query to process
- `k` (Optional[int]): Override for number of documents to retrieve

**Returns:**
- `AsyncIterator[StreamingChunk]`: Stream of response chunks

**Example:**
```python
async def stream_query():
    engine = RAGEngine()
    
    async for chunk in engine.query_stream("What are the benefits of renewable energy?"):
        print(chunk.content, end="")
        
        # Access metadata
        if chunk.metadata:
            cycle = chunk.metadata.get("cycle_number", 1)
            confidence = chunk.metadata.get("confidence_score", 0)
            print(f"\n[Cycle {cycle}, Confidence: {confidence:.2f}]")
        
        # Check for completion
        if chunk.is_complete:
            print("\n--- Response Complete ---")

asyncio.run(stream_query())
```

---

###### `get_engine_info() -> Dict`

Return engine configuration and statistics.

**Returns:**
- `Dict`: Engine configuration and statistics

**Example:**
```python
def get_info():
    engine = RAGEngine()
    info = engine.get_engine_info()
    
    print(f"Engine Type: {info['engine_type']}")
    print(f"Max Cycles: {info['max_reflexion_cycles']}")
    print(f"Confidence Threshold: {info['confidence_threshold']}")
    print(f"Memory Cache: {info['memory_cache_enabled']}")
    
    if 'memory_stats' in info:
        memory = info['memory_stats']
        print(f"Cache Size: {memory['size']}/{memory['max_size']}")
        print(f"Hit Rate: {memory['hit_rate']:.2%}")

get_info()
```

---

###### `clear_memory_cache() -> None`

Clear the memory cache.

**Example:**
```python
async def clear_cache():
    engine = RAGEngine()
    await engine.clear_memory_cache()
    print("Cache cleared successfully")

asyncio.run(clear_cache())
```

---

###### `count_documents() -> int`

Count documents in the vector store.

**Returns:**
- `int`: Number of documents in the vector store

**Example:**
```python
async def count_docs():
    engine = RAGEngine()
    count = await engine.count_documents()
    print(f"Total documents: {count}")

asyncio.run(count_docs())
```

---

###### `delete_all_documents(confirm_string: str = "CONFIRM") -> bool`

Delete all documents from the vector store.

**Parameters:**
- `confirm_string` (str): Must be "CONFIRM" (case-sensitive)

**Returns:**
- `bool`: True if deletion successful, False otherwise

**Example:**
```python
async def delete_all():
    engine = RAGEngine()
    success = await engine.delete_all_documents("CONFIRM")
    if success:
        print("All documents deleted successfully")
    else:
        print("Failed to delete documents")

asyncio.run(delete_all())
```

### Data Models

#### StreamingChunk

Represents a chunk of streaming response data.

**Attributes:**
- `content` (str): The text content of the chunk
- `is_complete` (bool): Whether this is the final chunk
- `metadata` (Optional[Dict]): Additional metadata about the chunk

**Metadata Fields:**
- `cycle_number` (int): Current reflexion cycle number
- `confidence_score` (float): Confidence score for the current response
- `is_cached` (bool): Whether the response was retrieved from cache
- `total_cycles` (int): Total number of cycles completed
- `total_processing_time` (float): Total processing time in seconds
- `final_confidence` (float): Final confidence score
- `reflexion_complete` (bool): Whether reflexion process completed

#### ReflexionEvaluation

Represents the evaluation of a response during reflexion.

**Attributes:**
- `confidence_score` (float): Confidence score (0.0-1.0)
- `decision` (ReflexionDecision): Next action decision
- `reasoning` (str): Explanation of the evaluation
- `follow_up_queries` (List[str]): Generated follow-up queries
- `covered_aspects` (List[str]): Aspects covered in the response
- `missing_aspects` (List[str]): Aspects missing from the response
- `uncertainty_phrases` (List[str]): Phrases indicating uncertainty

#### ReflexionDecision

Enumeration of possible reflexion decisions.

**Values:**
- `CONTINUE`: Continue with current approach
- `REFINE_QUERY`: Generate refined follow-up queries
- `COMPLETE`: Response is complete and satisfactory
- `INSUFFICIENT_DATA`: Knowledge base lacks necessary information

### Advanced Usage Examples

#### Batch Document Processing

```python
import asyncio
from pathlib import Path
from rag.src.rag.engine import RAGEngine

async def batch_ingest(directories):
    engine = RAGEngine()
    total_ingested = 0
    
    for directory in directories:
        if Path(directory).exists():
            count = await engine.ingest_documents(directory)
            total_ingested += count
            print(f"Ingested {count} documents from {directory}")
        else:
            print(f"Directory not found: {directory}")
    
    print(f"Total documents ingested: {total_ingested}")
    return total_ingested

# Usage
directories = ["./docs/technical", "./docs/research", "./docs/manuals"]
asyncio.run(batch_ingest(directories))
```

#### Custom Query Processing with Callbacks

```python
import asyncio
from typing import Callable
from rag.src.rag.engine import RAGEngine

async def process_with_callbacks(
    query: str,
    on_chunk: Callable[[str], None] = None,
    on_cycle_start: Callable[[int], None] = None,
    on_completion: Callable[[dict], None] = None
):
    engine = RAGEngine()
    current_cycle = 0
    response_text = ""
    
    async for chunk in engine.query_stream(query):
        # Handle new cycle
        if chunk.metadata and chunk.metadata.get("cycle_number", 1) != current_cycle:
            current_cycle = chunk.metadata["cycle_number"]
            if on_cycle_start:
                on_cycle_start(current_cycle)
        
        # Handle chunk content
        if chunk.content:
            response_text += chunk.content
            if on_chunk:
                on_chunk(chunk.content)
        
        # Handle completion
        if chunk.is_complete and chunk.metadata:
            if on_completion:
                on_completion(chunk.metadata)
    
    return response_text

# Usage with callbacks
def on_chunk_received(content):
    print(content, end="", flush=True)

def on_cycle_started(cycle_num):
    print(f"\n--- Starting Cycle {cycle_num} ---")

def on_processing_complete(metadata):
    print(f"\n--- Complete in {metadata.get('total_processing_time', 0):.2f}s ---")

async def main():
    response = await process_with_callbacks(
        "Explain quantum computing principles",
        on_chunk=on_chunk_received,
        on_cycle_start=on_cycle_started,
        on_completion=on_processing_complete
    )

asyncio.run(main())
```

#### Performance Monitoring

```python
import asyncio
import time
from rag.src.rag.engine import RAGEngine

class PerformanceMonitor:
    def __init__(self):
        self.queries = []
        self.start_time = None
    
    async def monitored_query(self, engine: RAGEngine, query: str):
        start_time = time.time()
        chunks_received = 0
        total_content_length = 0
        cycles = 0
        
        async for chunk in engine.query_stream(query):
            chunks_received += 1
            if chunk.content:
                total_content_length += len(chunk.content)
            
            if chunk.metadata and chunk.metadata.get("cycle_number"):
                cycles = max(cycles, chunk.metadata["cycle_number"])
        
        end_time = time.time()
        
        stats = {
            "query": query,
            "duration": end_time - start_time,
            "chunks_received": chunks_received,
            "content_length": total_content_length,
            "cycles": cycles,
            "timestamp": start_time
        }
        
        self.queries.append(stats)
        return stats
    
    def get_performance_summary(self):
        if not self.queries:
            return {"message": "No queries processed yet"}
        
        total_queries = len(self.queries)
        avg_duration = sum(q["duration"] for q in self.queries) / total_queries
        avg_cycles = sum(q["cycles"] for q in self.queries) / total_queries
        
        return {
            "total_queries": total_queries,
            "average_duration": avg_duration,
            "average_cycles": avg_cycles,
            "fastest_query": min(self.queries, key=lambda x: x["duration"]),
            "slowest_query": max(self.queries, key=lambda x: x["duration"])
        }

# Usage
async def performance_test():
    monitor = PerformanceMonitor()
    engine = RAGEngine()
    
    test_queries = [
        "What is machine learning?",
        "Compare different database types",
        "Explain the history and evolution of programming languages"
    ]
    
    for query in test_queries:
        print(f"Processing: {query}")
        stats = await monitor.monitored_query(engine, query)
        print(f"Completed in {stats['duration']:.2f}s using {stats['cycles']} cycles\n")
    
    summary = monitor.get_performance_summary()
    print("Performance Summary:")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Average duration: {summary['average_duration']:.2f}s")
    print(f"Average cycles: {summary['average_cycles']:.1f}")

asyncio.run(performance_test())
```

### Error Handling

#### Common Exceptions

```python
import asyncio
from rag.src.rag.engine import RAGEngine
from rag.src.exceptions import (
    RAGEngineError,
    DocumentIngestionError,
    VectorStoreError,
    LLMError,
    ConfigurationError
)

async def robust_query_processing():
    engine = RAGEngine()
    
    try:
        # Attempt to process query
        response_text = ""
        async for chunk in engine.query_stream("Complex technical query"):
            response_text += chunk.content
        
        return response_text
    
    except ConfigurationError as e:
        print(f"Configuration issue: {e}")
        print("Please check your .env file and configuration")
        return None
    
    except VectorStoreError as e:
        print(f"Vector store error: {e}")
        print("Check SurrealDB connection and credentials")
        return None
    
    except LLMError as e:
        print(f"LLM processing error: {e}")
        print("Check GitHub token and model availability")
        return None
    
    except DocumentIngestionError as e:
        print(f"Document processing error: {e}")
        print("Ensure documents are in supported formats")
        return None
    
    except RAGEngineError as e:
        print(f"General RAG engine error: {e}")
        return None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
response = asyncio.run(robust_query_processing())
if response:
    print("Query processed successfully")
else:
    print("Query processing failed")
```

#### Retry Logic

```python
import asyncio
from typing import Optional
from rag.src.rag.engine import RAGEngine

async def query_with_retry(
    query: str,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[str]:
    engine = RAGEngine()
    
    for attempt in range(max_retries + 1):
        try:
            response_text = ""
            async for chunk in engine.query_stream(query):
                response_text += chunk.content
            
            return response_text
        
        except Exception as e:
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries + 1} attempts failed. Last error: {e}")
                return None

# Usage
result = asyncio.run(query_with_retry("What is artificial intelligence?"))
if result:
    print("Query successful:", result[:100] + "...")
else:
    print("Query failed after all retries")
```

## Future REST API (Planned)

The following REST API endpoints are planned for future releases:

### Authentication

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "access_token": "jwt_token_here",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Document Management

#### Upload Documents

```http
POST /api/v1/documents/upload
Authorization: Bearer {token}
Content-Type: multipart/form-data

files: [file1.pdf, file2.txt, ...]
```

**Response:**
```json
{
  "uploaded_count": 5,
  "failed_count": 0,
  "total_documents": 150,
  "processing_time": 12.5
}
```

#### List Documents

```http
GET /api/v1/documents?page=1&limit=50&format=pdf
Authorization: Bearer {token}
```

**Response:**
```json
{
  "documents": [
    {
      "id": "doc_123",
      "filename": "example.pdf",
      "format": "pdf",
      "size": 1024576,
      "created_at": "2025-06-10T10:30:00Z",
      "chunk_count": 25
    }
  ],
  "total": 150,
  "page": 1,
  "limit": 50
}
```

#### Delete Documents

```http
DELETE /api/v1/documents/{document_id}
Authorization: Bearer {token}
```

### Query Processing

#### Standard Query

```http
POST /api/v1/query
Authorization: Bearer {token}
Content-Type: application/json

{
  "question": "What are the benefits of renewable energy?",
  "options": {
    "max_cycles": 3,
    "confidence_threshold": 0.8,
    "retrieval_k": 5
  }
}
```

**Response:**
```json
{
  "response": "Renewable energy offers numerous benefits...",
  "metadata": {
    "total_cycles": 2,
    "final_confidence": 0.85,
    "processing_time": 8.2,
    "sources_used": 12,
    "reflexion_complete": true
  }
}
```

#### Streaming Query

```http
GET /api/v1/query/stream?question=What%20is%20AI&stream=true
Authorization: Bearer {token}
Accept: text/event-stream
```

**Response (Server-Sent Events):**
```
data: {"content": "Artificial intelligence", "cycle": 1, "confidence": 0.6}

data: {"content": " refers to the development", "cycle": 1}

data: {"content": "...", "is_complete": true, "final_confidence": 0.87}
```

### System Management

#### Health Check

```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "vector_store": "connected",
    "llm_service": "available",
    "cache": "active"
  },
  "uptime": 86400
}
```

#### System Statistics

```http
GET /api/v1/stats
Authorization: Bearer {token}
```

**Response:**
```json
{
  "total_documents": 1500,
  "total_queries": 5000,
  "cache_hit_rate": 0.75,
  "average_response_time": 3.2,
  "reflexion_usage": 0.40,
  "storage_used": "2.5GB"
}
```

## SDK Configuration

### Environment-based Configuration

```python
import os
from rag.src.config.settings import Settings

# Override settings programmatically
settings = Settings(
    llm_model="meta/Meta-Llama-3.1-70B-Instruct",  # Use smaller model
    max_reflexion_cycles=3,  # Reduce cycles for faster responses
    confidence_threshold=0.7,  # Lower threshold
    enable_memory_cache=True,
    max_cache_size=500
)

# Use custom settings
from rag.src.rag.engine import RAGEngine
engine = RAGEngine(settings=settings)
```

### Custom Model Configuration

```python
from rag.src.llm.github_models import GitHubModelsLLM
from rag.src.embeddings.azure_ai import AzureAIEmbeddings
from rag.src.rag.engine import RAGEngine

# Configure custom LLM
custom_llm = GitHubModelsLLM(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    temperature=0.3,
    max_tokens=2000
)

# Configure custom embeddings
custom_embeddings = AzureAIEmbeddings(
    model_name="text-embedding-ada-002",
    batch_size=50
)

# Use custom components
engine = RAGEngine(
    llm=custom_llm,
    embeddings=custom_embeddings
)
```

## Best Practices

### 1. Query Optimization

```python
# Good: Specific, focused queries
"What are the key differences between SQL and NoSQL databases in terms of scalability and consistency?"

# Avoid: Vague, overly broad queries
"Tell me about databases"
```

### 2. Document Organization

```python
# Organize documents by topic for better retrieval
docs/
├── technical/
│   ├── databases/
│   ├── programming/
│   └── architecture/
├── business/
│   ├── strategy/
│   └── processes/
└── research/
    ├── papers/
    └── reports/
```

### 3. Performance Tuning

```python
# For faster responses (less comprehensive)
settings = Settings(
    max_reflexion_cycles=2,
    confidence_threshold=0.7,
    initial_retrieval_k=3
)

# For more comprehensive responses (slower)
settings = Settings(
    max_reflexion_cycles=5,
    confidence_threshold=0.85,
    initial_retrieval_k=5,
    reflexion_retrieval_k=7
)
```

### 4. Memory Management

```python
# Enable caching for repeated queries
settings = Settings(
    enable_memory_cache=True,
    max_cache_size=1000  # Adjust based on available RAM
)

# Clear cache periodically in long-running applications
async def periodic_cache_cleanup():
    while True:
        await asyncio.sleep(3600)  # Every hour
        await engine.clear_memory_cache()
        print("Cache cleared")
```

## Migration Guide

### From v0.x to v1.x

```python
# Old API (deprecated)
from rag import RAGSystem
system = RAGSystem()
response = system.query("What is AI?")

# New API
from rag.src.rag.engine import RAGEngine
import asyncio

async def main():
    engine = RAGEngine()
    response = ""
    async for chunk in engine.query_stream("What is AI?"):
        response += chunk.content
    return response

response = asyncio.run(main())
```

---

*API Documentation last updated: June 2025*