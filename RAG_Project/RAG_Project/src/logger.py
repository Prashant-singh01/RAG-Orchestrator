"""Application logging setup for the RAG project."""

from __future__ import annotations

import logging
from pathlib import Path

from config import LOG_PATH


def configure_logger(name: str = "rag_project") -> logging.Logger:
    """Create a shared logger with console and file handlers.

    Args:
        name: Logger name to use in the application.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_directory = Path(LOG_PATH)
    log_directory.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_directory / "rag_project.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured for the project."""
    return configure_logger(name)
