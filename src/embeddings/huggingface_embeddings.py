from typing import Any, List, cast

import torch
from sentence_transformers import SentenceTransformer

from ..config.settings import settings
from ..core.exceptions import EmbeddingException
from ..core.interfaces import EmbeddingInterface


class HuggingFaceEmbeddings(EmbeddingInterface):
    """HuggingFace embeddings with optimizations"""

    def __init__(self) -> None:
        super().__init__()

        try:
            self.model = SentenceTransformer(
                settings.embedding_model, device=settings.embedding_device
            )
            if hasattr(self.model, "encode"):
                encode_method = cast(Any, self.model.encode)
                self.model.encode = torch.compile(
                    encode_method, mode="reduce-overhead"
                )

        except Exception as e:
            raise EmbeddingException(f"Failed to load embedding model: {e}")

    async def embed_text(self, text: str) -> List[float]:
        """Embed single text"""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)[0]
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingException(f"Text embedding failed: {e}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently"""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                batch_size=32,  # optimizable later
                show_progress_bar=len(texts) > 100,
            )

            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise EmbeddingException(f"Document embedding failed: {e}")
