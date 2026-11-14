"""
Turns text into vectors (lists of numbers) with a local Ollama embedding model.

Texts with similar meaning get similar vectors, which is what makes searching
by meaning possible. Always use the SAME model for indexing and for questions.
"""
from __future__ import annotations

from typing import Optional

import ollama

from config import settings
from utils.helpers import OllamaError, call_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaEmbedder:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: Optional[int] = None,
        doc_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ) -> None:
        self.model = model or settings.EMBED_MODEL
        self.batch_size = max(1, batch_size or settings.EMBED_BATCH_SIZE)
        self.doc_prefix = settings.EMBED_DOC_PREFIX if doc_prefix is None else doc_prefix
        self.query_prefix = settings.EMBED_QUERY_PREFIX if query_prefix is None else query_prefix
        self._client = ollama.Client(host=base_url or settings.OLLAMA_BASE_URL, timeout=settings.OLLAMA_TIMEOUT)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = call_with_retry(
            lambda: self._client.embed(model=self.model, input=texts),
            what=f"embedding with '{self.model}'",
        )
        vectors = response["embeddings"]
        if len(vectors) != len(texts):
            raise OllamaError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts")
        return [list(v) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunks, in batches."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self.doc_prefix + t for t in texts[start : start + self.batch_size]]
            vectors.extend(self._embed(batch))
        logger.debug("Embedded %d texts with %s", len(texts), self.model)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single question."""
        return self._embed([self.query_prefix + text])[0]
