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


class DocumentProcessingException(Exception):
    """Exception raised for document processing operations."""

    pass
