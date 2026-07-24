"""Orchestrates the retrieval and answer generation workflow."""

from __future__ import annotations

from typing import Any

from src.llm_loader import LLMLoader
from src.logger import get_logger
from src.prompt_template import build_prompt
from src.retriever import Retriever


class RAGPipeline:
    """Main class that connects retrieval and LLM generation."""

    def __init__(self) -> None:
        """Initialize the retriever and LLM loader dependencies."""
        self.logger = get_logger(__name__)
        self.retriever = Retriever()
        self.llm_loader = LLMLoader()

    def ask(self, question: str) -> dict[str, Any]:
        """Answer a question using retrieved document context.

        Args:
            question: User input that should be answered from the vector store.

        Returns:
            Dictionary containing answer text and retrieved sources.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        self.logger.info("User question received: %s", question)

        retrieved_documents = self.retriever.retrieve(question)
        self.logger.info("Retrieved %s chunks for the query.", len(retrieved_documents))

        if not retrieved_documents:
            self.logger.warning("No context could be retrieved for the question.")
            return {
                "answer": "I couldn't find this information in the provided document.",
                "documents": [],
                "context": "",
            }

        context = "\n\n".join(document.page_content for document in retrieved_documents)
        prompt = build_prompt(context=context, question=question)
        answer = self.llm_loader.generate_response(prompt)

        cleaned_answer = answer.strip() if answer else ""
        if not cleaned_answer:
            cleaned_answer = "I couldn't find this information in the provided document."

        self.logger.info("LLM response generated successfully.")

        return {
            "answer": cleaned_answer,
            "documents": retrieved_documents,
            "context": context,
        }
