import uuid
from typing import Any, Dict, List

from surrealdb import AsyncSurreal

from ..config.settings import settings
from ..core.exceptions import VectorStoreException
from ..core.interfaces import Document, VectorStoreInterface
from ..embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from ..utils.logging import logger


class SurrealDBVectorStore(VectorStoreInterface):
    """SurrealDB implementation with native vector search"""

    def __init__(self):
        super().__init__()
        self.client = None
        self.connected = False
        self.embedding_function = HuggingFaceEmbeddings()

    async def _ensure_connection(self):
        """Ensure database connection and schema"""
        if self.connected and self.client is not None:
            return

        try:
            # Initialize the client
            self.client = AsyncSurreal(settings.surrealdb_url)

            # Connect and authenticate
            await self.client.connect(settings.surrealdb_url)
            await self.client.signin(
                {
                    "username": settings.surrealdb_user,
                    "password": settings.surrealdb_pass,
                }
            )
            await self.client.use(settings.surrealdb_ns, settings.surrealdb_db)

            # Ensure schema exists
            await self._setup_schema()
            self.connected = True
            logger.info("SurrealDB connection established")

        except Exception as e:
            logger.error(f"SurrealDB connection failed: {str(e)}")
            raise VectorStoreException(f"Connection failed: {str(e)}")

    async def _setup_schema(self):
        """Setup vector storage schema"""
        if not self.client:
            raise VectorStoreException("Client not initialized")

        schema_queries = [
            "DEFINE TABLE IF NOT EXISTS documents SCHEMAFULL;",
            "DEFINE FIELD IF NOT EXISTS content ON documents TYPE string;",
            "DEFINE FIELD IF NOT EXISTS metadata ON documents TYPE object;",
            "DEFINE FIELD IF NOT EXISTS embedding ON documents TYPE array<float>;",
            "DEFINE INDEX IF NOT EXISTS hnsw_embedding ON documents FIELDS embedding HNSW DIMENSION 384 DIST COSINE;",
        ]

        for query in schema_queries:
            try:
                await self.client.query(query)
            except Exception as e:
                logger.error(f"Schema setup error for query '{query}': {e}")
                raise VectorStoreException(f"Schema setup failed: {e}")

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents with embeddings to SurrealDB"""
        await self._ensure_connection()

        if not documents:
            return []

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            # Generate embeddings for all documents
            texts = [doc.content for doc in documents]
            embeddings = await self.embedding_function.embed_documents(texts)

            doc_ids = []

            # Batch insert documents
            for doc, embedding in zip(documents, embeddings):
                doc_id = doc.doc_id or str(uuid.uuid4())

                # Sanitize metadata
                clean_metadata = self._sanitize_metadata(doc.metadata)

                # Create document in SurrealDB
                await self.client.create(
                    "documents",
                    {
                        "id": doc_id,
                        "content": doc.content,
                        "metadata": clean_metadata,
                        "embedding": embedding,
                    },
                )

                doc_ids.append(doc_id)

            logger.info(f"Added {len(doc_ids)} documents to SurrealDB")
            return doc_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise VectorStoreException(f"Failed to add documents: {str(e)}")

    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform vector similarity search"""
        await self._ensure_connection()

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            # Generate query embedding
            query_embedding = await self.embedding_function.embed_text(query)

            # Perform similarity search using SurrealDB query
            results = await self.client.query(
                """
                SELECT id, content, metadata,
                       vector::similarity::cosine(embedding, $query_embedding) AS score
                FROM documents
                WHERE embedding IS NOT NONE
                ORDER BY score DESC
                LIMIT $k;
                """,
                {
                    "query_embedding": query_embedding,
                    "k": k,
                },
            )

            logger.debug(f"Raw SurrealDB results: {results}")

            # Check if we have results
            if not results or len(results) == 0:
                logger.warning("No results returned from SurrealDB")
                return []

            # Extract results from SurrealDB response
            search_results = results[0] if isinstance(results[0], list) else []

            if not search_results:
                logger.warning("Empty search results")
                return []

            documents = []
            for result in search_results:
                if not isinstance(result, dict):
                    continue

                doc = Document(
                    content=result.get("content", ""),
                    metadata={
                        **result.get("metadata", {}),
                        "similarity_score": result.get("score", 0.0),
                    },
                    doc_id=result.get("id"),
                )
                documents.append(doc)

            logger.info(f"Retrieved {len(documents)} documents from SurrealDB")
            return documents

        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []

    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        await self._ensure_connection()

        if not self.client:
            raise VectorStoreException("Client not connected")

        try:
            if doc_ids:
                for doc_id in doc_ids:
                    await self.client.delete(f"documents:{doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise VectorStoreException(f"Failed to delete documents: {str(e)}")

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for SurrealDB storage"""
        if not metadata:
            return {}

        sanitized = {}
        for k, v in metadata.items():
            if v is None:
                sanitized[str(k)] = ""
            elif isinstance(v, (str, int, float, bool)):
                sanitized[str(k)] = v
            else:
                sanitized[str(k)] = str(v)

        return sanitized

    async def close(self):
        """Close database connection"""
        if self.client:
            await self.client.close()
            self.connected = False
            logger.info("SurrealDB connection closed")
