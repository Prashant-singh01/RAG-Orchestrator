"""Loads PDF documents into LangChain document objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader

from src.logger import get_logger


class DocumentLoader:
    """Wrapper around the PDF loader used to read uploaded documents."""

    def __init__(self, pdf_path: str) -> None:
        """Resolve a PDF path independently of the caller's working directory."""
        self.logger = get_logger(__name__)
        base_dir = Path(__file__).resolve().parent.parent
        resolved_path = Path(pdf_path)

        if resolved_path.is_absolute():
            self.pdf_path = str(resolved_path)
        else:
            self.pdf_path = str((base_dir / resolved_path).resolve())

    def load_documents(self) -> list[Any]:
        """Load the PDF file and return LangChain document chunks."""
        try:
            self.logger.info("Loading PDF from %s", self.pdf_path)
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            self.logger.info("Loaded %s pages from the PDF.", len(documents))
            return documents
        except FileNotFoundError as exc:
            self.logger.exception("PDF file not found: %s", self.pdf_path)
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}") from exc
        except Exception as exc:
            self.logger.exception("Failed to load PDF document.")
            raise RuntimeError(f"Failed to load PDF document: {exc}") from exc
