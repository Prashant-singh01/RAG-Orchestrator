"""Simple evaluation helpers for the RAG workflow."""

from __future__ import annotations

import time
from typing import Any


def evaluate_response(answer: str, context: str, start_time: float) -> dict[str, Any]:
    """Calculate lightweight response metrics.

    Args:
        answer: Generated answer from the LLM.
        context: Context string that was supplied to the model.
        start_time: Time captured before the LLM call begins.

    Returns:
        Dictionary containing evaluation metrics.
    """
    response_time = round(time.time() - start_time, 4)
    return {
        "retrieved_chunks_count": len(context.split("\n\n")) if context else 0,
        "answer_length": len(answer.strip()),
        "response_time_seconds": response_time,
        "context_exists": bool(context.strip()),
    }


def evaluate_retrieval(documents: list[Any]) -> dict[str, Any]:
    """Evaluate retrieval quality with a simple count metric.

    Args:
        documents: Retrieved document chunks from the vector store.

    Returns:
        Retrieval metrics dictionary.
    """
    return {
        "retrieved_chunks_count": len(documents),
        "context_exists": bool(documents),
    }
