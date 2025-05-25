from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import SimpleDirectoryReader

from ..core.exceptions import RAGException
from ..core.interfaces import Document


class DocumentLoader:
    """Document loader with async support"""

    def __init__(self):
        self.supported_extensions = {".txt", ".md", ".pdf", ".docs", ".html"}

    async def load_from_directory(self, directory_path: str) -> List[Document]:
        """Load documents from directory asynchronously"""
        try:
            path = Path(directory_path)
            if not path.exists():
                raise RAGException(f"Directory not found: {directory_path}")
            reader = SimpleDirectoryReader(
                input_dir=str(path),
                recursive=True,
                required_exts=list(self.supported_extensions),
            )

            # Load Documents
            llama_docs = reader.load_data()

            # Convert to our document format
            documents = []
            for llama_doc in llama_docs:
                doc = Document(
                    content=llama_doc.text,
                    metadata={
                        "source": str(llama_doc.metadata.get("file_path", "")),
                        "file_name": llama_doc.metadata.get("file_name", ""),
                        "file_type": llama_doc.metadata.get("file_type", ""),
                        **llama_doc.metadata,
                    },
                )
                documents.append(doc)
            return documents
        except Exception as e:
            raise RAGException(f"Failed to Load Documents: {e}")

    async def load_from_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Load a single document from text"""
        return Document(content=text, metadata=metadata or {})
