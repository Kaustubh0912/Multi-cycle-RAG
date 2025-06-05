from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


class ReflexionDecision(Enum):
    """Reflexion loop decision types"""

    CONTINUE = "continue"
    REFINE_QUERY = "refine_query"
    COMPLETE = "complete"
    INSUFFICIENT_DATA = "insufficient_data"

    def __str__(self) -> str:
        """String representation of the decision"""
        return self.value


@dataclass
class Document:
    """Standard document format across the system"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None


@dataclass
class StreamingChunk:
    """Streaming Response Chunk"""

    content: str
    is_complete: bool = False
    usage_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReflexionEvaluation:
    """Evaluation result from reflexion assessment"""

    confidence_score: float
    decision: ReflexionDecision
    reasoning: str
    follow_up_queries: List[str]
    covered_aspects: List[str]
    missing_aspects: List[str]
    uncertainty_phrases: List[str]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReflexionCycle:
    """Single reflexion cycle data"""

    cycle_number: int
    query: str
    retrieved_docs: List[Document]
    partial_answer: str
    evaluation: ReflexionEvaluation
    timestamp: datetime
    processing_time: float

    def __post_init__(self):
        if not hasattr(self, "timestamp"):
            self.timestamp = datetime.now()


@dataclass
class ReflexionMemory:
    """Memory cache for reflexion loops"""

    original_query: str
    cycles: List[ReflexionCycle] = field(default_factory=list)
    final_answer: str = ""
    total_processing_time: float = 0.0
    total_documents_retrieved: int = 0

    def add_cycle(self, cycle: ReflexionCycle):
        """Add a reflexion cycle to memory"""
        self.cycles.append(cycle)
        self.total_processing_time += cycle.processing_time
        self.total_documents_retrieved += len(cycle.retrieved_docs)

    def get_all_partial_answers(self) -> List[str]:
        """Get all partial answers from cycles"""
        return [cycle.partial_answer for cycle in self.cycles if cycle.partial_answer]

    def get_all_retrieved_docs(self) -> List[Document]:
        """Get all unique retrieved documents"""
        all_docs = []
        seen_ids = set()
        for cycle in self.cycles:
            for doc in cycle.retrieved_docs:
                if doc.doc_id and doc.doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc.doc_id)
        return all_docs


class LLMInterface(ABC):
    """Abstract base class for all LLM implementations"""

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


class ReflexionEvaluatorInterface(ABC):
    """Abstract interface for reflexion evaluation"""

    @abstractmethod
    async def evaluate_response(
        self,
        query: str,
        partial_answer: str,
        retrieved_docs: List[Document],
        cycle_number: int,
    ) -> ReflexionEvaluation:
        """Evaluate if response sufficiently answers the query"""
        pass

    @abstractmethod
    async def generate_follow_up_queries(
        self,
        original_query: str,
        partial_answer: str,
        missing_aspects: List[str],
    ) -> List[str]:
        """Generate follow-up queries based on missing aspects"""
        pass


class VectorStoreInterface(ABC):
    """Abstract base class for all vector store implementations"""

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
