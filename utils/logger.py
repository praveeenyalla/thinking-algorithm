"""Logger setup for the CLI pipeline."""

from __future__ import annotations

import logging

from config import get_settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance."""
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
    return logging.getLogger(name)
