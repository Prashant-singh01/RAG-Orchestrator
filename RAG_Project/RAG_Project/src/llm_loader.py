"""Loads and runs the Hugging Face generation model."""

from __future__ import annotations

from threading import Thread
from typing import Iterator

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextIteratorStreamer

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

    def _build_generation_kwargs(self, prompt: str, streamer: TextIteratorStreamer | None = None) -> dict[str, object]:
        """Construct the generation kwargs used by the Hugging Face model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return {
            **inputs,
            "max_new_tokens": LLM_MAX_NEW_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "num_beams": 4,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

    def stream_response(self, prompt: str) -> Iterator[str]:
        """Yield LLM tokens incrementally so the Streamlit UI can render them live."""
        try:
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generation_kwargs = self._build_generation_kwargs(prompt, streamer=streamer)

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for token in streamer:
                if token:
                    yield token

            thread.join()
        except Exception as exc:
            self.logger.exception("Failed to stream LLM response.")
            raise RuntimeError(f"Failed to stream LLM response: {exc}") from exc

    def generate_response(self, prompt: str) -> str:
        """Generate a grounded answer from the supplied prompt."""
        try:
            return "".join(self.stream_response(prompt))
        except Exception as exc:
            self.logger.exception("Failed to generate LLM response.")
            raise RuntimeError(f"Failed to generate LLM response: {exc}") from exc
