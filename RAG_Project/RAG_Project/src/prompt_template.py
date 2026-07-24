"""Reusable prompt builder for the RAG pipeline."""

from __future__ import annotations

from config import PROMPT_ROLE, PROMPT_TASK, PROMPT_INSTRUCTIONS, PROMPT_EXPECTED_ANSWER


def build_prompt(context: str, question: str) -> str:
    """Construct a controlled prompt to reduce hallucinations.

    Args:
        context: Retrieved document context from the vector store.
        question: User question that should be answered.

    Returns:
        Prompt string for the language model.
    """
    return (
        f"Role: {PROMPT_ROLE}\n"
        f"Task: {PROMPT_TASK}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Instructions:\n{PROMPT_INSTRUCTIONS}\n\n"
        f"Expected Answer:\n{PROMPT_EXPECTED_ANSWER}"
    )
