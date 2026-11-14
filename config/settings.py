"""
Central configuration for the whole project.

Every other module imports its settings from here, so no path, model name or
number is hard-coded anywhere else. Values come from the .env file in the
project root; sensible defaults are used when a value is missing.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------------------------------- #
# Small helpers to read typed values from the environment
# --------------------------------------------------------------------------- #
def _str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    """Relative paths in .env are resolved against the project root."""
    p = Path(os.getenv(name, default)).expanduser()
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PDF_DIR: Path = _path("PDF_DIR", "data/pdfs")
VECTOR_DIR: Path = _path("VECTOR_DIR", "data/vector_store")
LOG_DIR: Path = _path("LOG_DIR", "logs")
MANIFEST_PATH: Path = VECTOR_DIR / "manifest.json"
COLLECTION_NAME: str = _str("COLLECTION_NAME", "pdf_chunks")

# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
OLLAMA_BASE_URL: str = _str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT: int = _int("OLLAMA_TIMEOUT", 120)  # seconds per request

# Embeddings
EMBED_MODEL: str = _str("EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE: int = _int("EMBED_BATCH_SIZE", 32)
# Some embedding models work better with task prefixes
# (e.g. nomic-embed-text: "search_document: " / "search_query: ").
EMBED_DOC_PREFIX: str = os.getenv("EMBED_DOC_PREFIX", "")
EMBED_QUERY_PREFIX: str = os.getenv("EMBED_QUERY_PREFIX", "")

# Chat model
LLM_MODEL: str = _str("LLM_MODEL", "llama3.1:8b")
TEMPERATURE: float = _float("TEMPERATURE", 0.1)
NUM_CTX: int = _int("NUM_CTX", 8192)  # model context window in tokens

# --------------------------------------------------------------------------- #
# Chunking (sizes are in characters; ~4 characters ≈ 1 token for English)
# --------------------------------------------------------------------------- #
CHUNK_SIZE: int = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 150)
MIN_CHUNK_CHARS: int = _int("MIN_CHUNK_CHARS", 40)

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
TOP_K: int = _int("TOP_K", 4)                 # chunks given to the LLM
FETCH_K: int = _int("FETCH_K", 12)            # chunks fetched before reranking
SCORE_THRESHOLD: float = _float("SCORE_THRESHOLD", 0.3)  # min cosine similarity
USE_RERANKER: bool = _bool("USE_RERANKER", False)

# --------------------------------------------------------------------------- #
# Chat memory & logging
# --------------------------------------------------------------------------- #
MAX_HISTORY_TURNS: int = _int("MAX_HISTORY_TURNS", 3)
LOG_LEVEL: str = _str("LOG_LEVEL", "INFO").upper()


def ensure_dirs() -> None:
    """Create data and log folders if they do not exist yet."""
    for folder in (PDF_DIR, VECTOR_DIR, LOG_DIR):
        folder.mkdir(parents=True, exist_ok=True)


ensure_dirs()
