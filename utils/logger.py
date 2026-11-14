"""
Central logging setup.

Usage in any module:
    from utils.logger import get_logger
    logger = get_logger(__name__)

All loggers are children of the "rag" logger, which writes to the console and
to logs/app.log (rotated so the file never grows forever). Streamlit re-runs
the script many times, so handlers are only attached once.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import settings

_ROOT_NAME = "rag"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _configure_root() -> logging.Logger:
    root = logging.getLogger(_ROOT_NAME)
    if root.handlers:  # already configured (e.g. Streamlit re-run)
        return root

    root.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.LOG_DIR / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a logger that shares the project's handlers and format."""
    _configure_root()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
