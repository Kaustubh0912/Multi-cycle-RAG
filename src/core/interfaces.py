from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from typing_extensions import AsyncIterator


@dataclass
class Document:
    """Standard document format across the system"""

    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None


@dataclass
class QueryResult:
    "Standard query result format"

    response: str
    source_documents: List[Document]
    metadata: Dict[str, Any]


@dataclass
class StreamingChunk:
    """Streaming Response Chunk"""

    content: str
    is_complete: bool = False
    usage_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMInterface(ABC):
    """Abstract base class for all LLM implementations"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        pass

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Chat-style interaction"""
        pass

    @abstractmethod
    def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Generate text from prompt with streaming"""
        pass

    @abstractmethod
    def chat_stream(
        self, messages: list[Dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Chat-style interaction with streaming"""
        pass


class VectorStoreInterface(ABC):
    """ABstract base class for all vector store implementations"""

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents and return their IDs"""
        pass

    @abstractmethod
    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        pass

    @abstractmethod
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass


class EmbeddingInterface(ABC):
    """Abstract base class for embedding models"""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        pass
