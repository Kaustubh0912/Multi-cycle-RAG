import uuid
from typing import Any, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from ..config.settings import settings
from ..core.exceptions import VectorStoreException
from ..core.interfaces import Document, VectorStoreInterface
from ..embeddings.huggingface_embeddings import HuggingFaceEmbeddings


class ChromaVectorStore(VectorStoreInterface):
    """ChromaDB implementation with async patterns"""

    def __init__(self):
        super().__init__()
        try:
            # Initialize ChromaDB with persistence
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False, allow_reset=True
                ),
            )

            # Initialize embedding function
            self.embedding_function = HuggingFaceEmbeddings()

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},  # similarity metric
            )

        except Exception as e:
            raise VectorStoreException(f"Failed to initialize ChromaDB: {e}")

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add Documents with batch processing"""
        try:
            doc_ids = []
            texts = []
            metadatas = []

            for doc in documents:
                doc_id = doc.doc_id or str(uuid.uuid4())
                doc_ids.append(doc_id)
                texts.append(doc.content)
                metadatas.append(doc.metadata)

            # Generate embeddings efficiently
            embeddings = await self.embedding_function.embed_documents(texts)

            # Fix 1: Convert embeddings to numpy arrays if needed
            # ChromaDB can be picky about embedding formats
            processed_embeddings = []
            for embedding in embeddings:
                if isinstance(embedding, list):
                    processed_embeddings.append(
                        np.array(embedding, dtype=np.float32)
                    )
                else:
                    processed_embeddings.append(embedding)

            self.collection.add(
                documents=texts,
                embeddings=processed_embeddings,  # Use processed embeddings
                metadatas=metadatas,
                ids=doc_ids,
            )

            return doc_ids
        except Exception as e:
            raise VectorStoreException(f"Failed to add Documents: {e}")

    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Similarity search with query optimization and proper error handling"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_function.embed_text(query)

            # Convert to numpy array if needed
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)

            # Perform Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            # Fix 2-6: Add proper null checks before accessing results
            if not results or not isinstance(results, dict):
                return []

            documents_list = results.get("documents")
            metadatas_list = results.get("metadatas")
            distances_list = results.get("distances")
            ids_list = results.get("ids")

            # Check if any required fields are None or empty
            if not documents_list or not documents_list[0]:
                return []

            # Convert to Document Objects with proper null handling
            documents = []
            doc_count = len(documents_list[0])

            for i in range(doc_count):
                # Safely access documents
                content = (
                    documents_list[0][i]
                    if documents_list and documents_list[0]
                    else ""
                )

                # Safely access metadata with type conversion
                metadata_raw = None
                if (
                    metadatas_list
                    and metadatas_list[0]
                    and i < len(metadatas_list[0])
                ):
                    metadata_raw = metadatas_list[0][i]

                # Convert metadata to proper Dict[str, Any] format
                metadata: dict[str, Any] = {}
                if metadata_raw:
                    # Handle ChromaDB metadata format conversion
                    if isinstance(metadata_raw, dict):
                        metadata = {str(k): v for k, v in metadata_raw.items()}
                    else:
                        metadata = {}

                # Safely access document ID
                doc_id: Optional[str] = None
                if ids_list and ids_list[0] and i < len(ids_list[0]):
                    doc_id = ids_list[0][i]

                # Safely access distance and calculate similarity score
                similarity_score = 0.0
                if (
                    distances_list
                    and distances_list[0]
                    and i < len(distances_list[0])
                ):
                    distance = distances_list[0][i]
                    similarity_score = (
                        1 - distance if distance is not None else 0.0
                    )

                metadata["similarity_score"] = similarity_score

                doc = Document(
                    content=content,
                    metadata=metadata,
                    doc_id=doc_id,
                )
                documents.append(doc)

            return documents

        except Exception as e:
            raise VectorStoreException(
                f"Failed to perform similarity search: {e}"
            )

    async def delete_documents(self, doc_ids: List[str]):
        """Delete documents by IDs"""
        try:
            if doc_ids:
                self.collection.delete(ids=doc_ids)
        except Exception as e:
            raise VectorStoreException(f"Failed to delete documents: {e}")
