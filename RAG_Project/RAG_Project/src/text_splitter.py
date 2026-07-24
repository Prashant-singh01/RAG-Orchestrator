"""Split loaded documents into retrieval-friendly text chunks."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from src.logger import get_logger


class TextChunker:
    """Create evenly sized text chunks with overlap for retrieval."""

    def __init__(self) -> None:
        """Initialize the recursive character splitter with project settings."""
        self.logger = get_logger(__name__)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def split_documents(self, documents: list[Any]) -> list[Any]:
        """Split the input LangChain documents into chunk objects."""
        try:
            chunks = self.splitter.split_documents(documents)
            self.logger.info("Generated %s text chunks.", len(chunks))
            return chunks
        except Exception as exc:
            self.logger.exception("Failed to split documents into chunks.")
            raise RuntimeError(f"Failed to split documents into chunks: {exc}") from exc

