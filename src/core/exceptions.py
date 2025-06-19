# rag\src\core\exceptions.py
class RAGException(Exception):
    """Base exception for RAG system"""

    pass


class LLMException(RAGException):
    """LLM-related exceptions"""

    pass


class VectorStoreException(RAGException):
    """Vector store-related exceptions"""

    pass


class EmbeddingException(RAGException):
    """Embedding-related exceptions"""

    pass


class DocumentProcessingException(RAGException):
    """Exception raised for document processing operations."""

    pass


class WebSearchException(RAGException):
    """Web search-related exceptions"""

    pass


class WebSearchConfigurationException(WebSearchException):
    """Web search configuration exceptions"""

    pass


class WebSearchAPIException(WebSearchException):
    """Web search API exceptions"""

    pass


class ContentExtractionException(WebSearchException):
    """Content extraction exceptions"""

    pass
