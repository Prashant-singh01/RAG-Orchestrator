"""Loads and runs the Hugging Face generation model."""

from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import (
    FALLBACK_LLM_MODEL,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PREFERRED_LLM_MODEL,
)
from src.logger import get_logger


class LLMLoader:
    """Small wrapper around the text-to-text Hugging Face model."""

    def __init__(self) -> None:
        """Load the tokenizer and model on initialization."""
        self.logger = get_logger(__name__)
        self.model_name = LLM_MODEL
        try:
            self._load_model(self.model_name)
        except Exception as exc:
            self.logger.warning(
                "Preferred model %s is unavailable (%s); falling back to %s.",
                PREFERRED_LLM_MODEL,
                exc,
                FALLBACK_LLM_MODEL,
            )
            self.model_name = FALLBACK_LLM_MODEL
            self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> None:
        """Load the tokenizer and the generation model for a given model name."""
        try:
            self.logger.info("Loading tokenizer for %s", model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info("Loading model for %s", model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.logger.info("Model loaded successfully for %s.", model_name)
        except Exception as exc:
            self.logger.exception("Failed to initialize LLM model: %s", model_name)
            raise RuntimeError(f"Failed to initialize LLM model: {exc}") from exc

    def generate_response(self, prompt: str) -> str:
        """Generate a grounded answer from the supplied prompt."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                num_beams=4,
                do_sample=False,
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as exc:
            self.logger.exception("Failed to generate LLM response.")
            raise RuntimeError(f"Failed to generate LLM response: {exc}") from exc
