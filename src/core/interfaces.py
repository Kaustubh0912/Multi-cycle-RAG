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


@dataclass
class SubQuery:
    """Individual sub-query with it's answer"""

    question: str
    answer: str
    source_documents: Optional[List[Document]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.source_documents is None:
            self.source_documents = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DecomposedQuery:
    """Decomposed query with sub-queries and final reasoning"""

    original_question: str
    sub_queries: List[SubQuery]
    final_answer: str = ""
    reasoning_steps: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.reasoning_steps is None:
            self.reasoning_steps = []
        if self.metadata is None:
            self.metadata = {}


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


class QueryDecomposerInterface(ABC):
    """Abstract interface for query decomposition"""

    @abstractmethod
    async def decompose_query(
        self, query: str, context: Optional[str] = None
    ) -> List[str]:
        """Decompose a complex query into sub-queries"""
        pass

    @abstractmethod
    async def should_decompose(self, query: str) -> bool:
        """Determine if a query needs decomposition"""
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
