"""Persist and load a Chroma vector store for document retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from src.logger import get_logger


class VectorStore:
    """Manage local embedding-backed vector storage."""

    def __init__(self) -> None:
        """Initialize embeddings and prepare the database path."""
        self.logger = get_logger(__name__)
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.db: Any | None = None
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

    def create_vector_store(self, chunks: list[Any]) -> Any:
        """Create or refresh the persisted Chroma collection from document chunks."""
        try:
            self.logger.info("Creating vector store with %s chunks.", len(chunks))
            self.db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=CHROMA_DB_PATH,
                collection_name=COLLECTION_NAME,
            )
            self.logger.info("Vector store created successfully.")
            return self.db
        except Exception as exc:
            self.logger.exception("Failed to create vector store.")
            raise RuntimeError(f"Failed to create vector store: {exc}") from exc

    def load_vector(self) -> Any:
        """Load the persisted Chroma collection for similarity retrieval."""
        try:
            self.logger.info("Loading existing vector store from %s", CHROMA_DB_PATH)
            self.db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embedding_model,
                collection_name=COLLECTION_NAME,
            )
            self.logger.info("Vector store loaded successfully.")
            return self.db
        except Exception as exc:
            self.logger.exception("Failed to load vector store.")
            raise RuntimeError(f"Failed to load vector store: {exc}") from exc