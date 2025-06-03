from typing import AsyncIterator, Optional

from ..config.settings import settings
from ..core.interfaces import LLMInterface, StreamingChunk, VectorStoreInterface
from .reflexion_engine import ReflexionRAGEngine


class AdvancedRAGEngine:
    """Reflexion-based RAG engine"""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        vector_store: Optional[VectorStoreInterface] = None,
    ):
        self.reflexion_engine = ReflexionRAGEngine(
            generation_llm=llm, vector_store=vector_store
        )
        self.vector_store = self.reflexion_engine.vector_store
        self.document_loader = self.reflexion_engine.document_loader
        self.document_processor = self.reflexion_engine.document_processor

    async def ingest_documents(self, directory_path: str) -> int:
        """Ingest documents from directory"""
        return await self.reflexion_engine.ingest_documents(directory_path)

    async def query_stream(
        self, question: str, k: Optional[int] = None
    ) -> AsyncIterator[StreamingChunk]:
        """Process query using reflexion architecture"""
        async for chunk in self.reflexion_engine.query_with_reflexion_stream(
            question
        ):
            yield chunk

    def get_engine_info(self) -> dict:
        """Return engine configuration information"""
        return {
            "engine_type": "ReflexionRAGEngine",
            "max_reflexion_cycles": settings.max_reflexion_cycles,
            "confidence_threshold": settings.confidence_threshold,
            "memory_cache_enabled": settings.enable_memory_cache,
            "memory_stats": self.reflexion_engine.get_memory_stats(),
        }


# Backward compatibility
RAGEngine = AdvancedRAGEngine
