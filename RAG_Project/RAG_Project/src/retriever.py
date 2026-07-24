"""Retrieve top-k matching chunks from the vector store."""

from __future__ import annotations

from typing import Any

from config import TOP_K
from src.logger import get_logger
from src.vector_store import VectorStore


class Retriever:
    """Thin wrapper around the Chroma retrieval API."""

    def __init__(self) -> None:
        """Initialize the vector store connection used for retrieval."""
        self.logger = get_logger(__name__)
        self.vector_store = VectorStore()
        self.db = self.vector_store.load_vector()

    def retrieve(self, query: str) -> list[Any]:
        """Return the most relevant chunks for the given query."""
        try:
            self.logger.info("Retrieving top %s chunks for query: %s", TOP_K, query)
            results = self.db.similarity_search(query=query, k=TOP_K)
            self.logger.info("Retrieved %s chunks.", len(results))
            return results
        except Exception as exc:
            self.logger.exception("Failed to retrieve context.")
            raise RuntimeError(f"Failed to retrieve context: {exc}") from exc
