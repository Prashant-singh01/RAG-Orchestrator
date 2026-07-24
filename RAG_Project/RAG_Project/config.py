"""Central configuration values for the RAG project."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
LOG_PATH = str(BASE_DIR / "logs")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PREFERRED_LLM_MODEL = "google/flan-t5-large"
FALLBACK_LLM_MODEL = "google/flan-t5-base"
LLM_MODEL = PREFERRED_LLM_MODEL
LLM_MAX_NEW_TOKENS = 256
LLM_TEMPERATURE = 0.2

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

COLLECTION_NAME = "pdf_collection"

PROMPT_ROLE = "You are a careful and evidence-based assistant."
PROMPT_TASK = "Answer the user's question using only the supplied context."
PROMPT_INSTRUCTIONS = (
    "Use only the supplied context. "
    "If the context does not contain enough information to answer safely, "
    "reply with: 'I couldn't find this information in the provided document.'"
)
PROMPT_EXPECTED_ANSWER = (
    "A concise answer grounded in the provided context only. "
    "Do not invent missing information."
)
