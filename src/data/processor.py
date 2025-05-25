from typing import List

from ..config.settings import settings
from ..core.interfaces import Document


class CustomTextSplitter:
    """Custom text splitter with recursive character splitting"""

    def __init__(
        self, chunk_size: int, chunk_overlap: int, separators: List[str]
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """Split text recursively using different separators"""

        def _split_text_recursive(
            text: str, separators: List[str]
        ) -> List[str]:
            if len(text) <= self.chunk_size:
                return [text] if text.strip() else []

            if not separators:
                # No more separators, force split by character count
                return [
                    text[i : i + self.chunk_size]
                    for i in range(
                        0, len(text), self.chunk_size - self.chunk_overlap
                    )
                ]

            separator = separators[0]
            remaining_separators = separators[1:]

            if separator == "":
                # Character-level splitting
                return [
                    text[i : i + self.chunk_size]
                    for i in range(
                        0, len(text), self.chunk_size - self.chunk_overlap
                    )
                ]

            parts = text.split(separator)
            result = []
            current_chunk = ""

            for part in parts:
                if len(current_chunk + separator + part) <= self.chunk_size:
                    current_chunk += (separator if current_chunk else "") + part
                else:
                    if current_chunk:
                        result.append(current_chunk)

                    if len(part) > self.chunk_size:
                        # Part is too long, recursively split it
                        result.extend(
                            _split_text_recursive(part, remaining_separators)
                        )
                        current_chunk = ""
                    else:
                        current_chunk = part

            if current_chunk:
                result.append(current_chunk)

            return result

        return _split_text_recursive(text, self.separators)


class DocumentProcessor:
    """Document processing with intelligent chunking"""

    def __init__(self):
        self.text_splitter = CustomTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def process_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """Process documents with intelligent chunking"""
        processed_docs = []
        for doc in documents:
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc.content)

            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    # 🔥 CRITICAL FIX: Ensure no None values in metadata
                    chunk_metadata = {}

                    # Copy original metadata, replacing None values
                    for k, v in doc.metadata.items():
                        if v is not None:
                            chunk_metadata[k] = v
                        else:
                            chunk_metadata[k] = (
                                ""  # Replace None with empty string
                            )

                    # Add chunk-specific metadata
                    chunk_metadata.update(
                        {
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "original_doc_id": doc.doc_id
                            or "",  # Ensure not None
                        }
                    )

                    chunk_doc = Document(
                        content=chunk.strip(),
                        metadata=chunk_metadata,
                    )
                    processed_docs.append(chunk_doc)
        return processed_docs
