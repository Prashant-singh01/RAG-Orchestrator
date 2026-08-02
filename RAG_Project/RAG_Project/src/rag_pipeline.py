"""Orchestrates the retrieval and answer generation workflow."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from src.llm_loader import LLMLoader
from src.logger import get_logger
from src.prompt_template import build_prompt, contains_prompt_injection_attempt
from src.retriever import Retriever


class RAGPipeline:
    """Main class that connects retrieval and LLM generation."""

    def __init__(self) -> None:
        """Initialize the retriever and LLM loader dependencies."""
        self.logger = get_logger(__name__)
        self.retriever = Retriever()
        self.llm_loader = LLMLoader()

    def prepare_question(self, question: str) -> dict[str, Any]:
        """Retrieve context and construct the grounded prompt for a user question."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        self.logger.info("User question received: %s", normalized_question)

        retrieved_documents = self.retriever.retrieve(normalized_question)
        self.logger.info("Retrieved %s chunks for the query.", len(retrieved_documents))

        if not retrieved_documents:
            self.logger.warning("No context could be retrieved for the question.")
            return {
                "answer": "I couldn't find this information in the provided document.",
                "documents": [],
                "context": "",
                "prompt": "",
                "question": normalized_question,
                "prompt_injection_detected": contains_prompt_injection_attempt(normalized_question),
            }

        context = "\n\n".join(document.page_content for document in retrieved_documents)
        prompt = build_prompt(context=context, question=normalized_question)
        return {
            "answer": "",
            "documents": retrieved_documents,
            "context": context,
            "prompt": prompt,
            "question": normalized_question,
            "prompt_injection_detected": contains_prompt_injection_attempt(normalized_question),
        }

    def ask(self, question: str) -> dict[str, Any]:
        """Answer a question using retrieved document context."""
        prepared = self.prepare_question(question)
        if not prepared["documents"]:
            return {
                "answer": prepared["answer"],
                "documents": [],
                "context": "",
                "question": prepared["question"],
                "prompt_injection_detected": prepared["prompt_injection_detected"],
            }

        answer = self.llm_loader.generate_response(prepared["prompt"])
        cleaned_answer = answer.strip() if answer else ""
        if not cleaned_answer:
            cleaned_answer = "I couldn't find this information in the provided document."

        self.logger.info("LLM response generated successfully.")

        return {
            "answer": cleaned_answer,
            "documents": prepared["documents"],
            "context": prepared["context"],
            "question": prepared["question"],
            "prompt_injection_detected": prepared["prompt_injection_detected"],
        }

    def stream_answer(self, prepared: dict[str, Any]) -> Iterator[str]:
        """Stream the final answer token-by-token after retrieval."""
        if not prepared["documents"]:
            yield prepared["answer"]
            return

        self.logger.info("Streaming answer for question: %s", prepared["question"])
        yield from self.llm_loader.stream_response(prepared["prompt"])
