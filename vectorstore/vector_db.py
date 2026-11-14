"""
The "library shelf": stores chunks with their vectors and finds the closest
ones to a question.

All Chroma-specific code lives here. To switch to FAISS or another store, only
this file needs to change (keep the same method names).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from ingestion.chunker import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)

_ADD_BATCH = 1000


class VectorDB:
    def __init__(self, path: Optional[Path] = None, collection_name: Optional[str] = None) -> None:
        self.path = Path(path or settings.VECTOR_DIR)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self._client = chromadb.PersistentClient(
            path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self._open_collection()

    def _open_collection(self):
        # We always pass our own vectors, so Chroma's built-in embedding function is disabled.
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    # ------------------------------------------------------------------ write
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for start in range(0, len(chunks), _ADD_BATCH):
            part = chunks[start : start + _ADD_BATCH]
            self.collection.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                metadatas=[_clean_metadata(c.metadata) for c in part],
                embeddings=[list(e) for e in embeddings[start : start + _ADD_BATCH]],
            )

    def delete_source(self, source: str) -> None:
        """Remove every chunk that came from one PDF."""
        self.collection.delete(where={"source": source})

    def reset(self) -> None:
        """Delete everything and start with an empty collection."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # collection may not exist yet
            pass
        self.collection = self._open_collection()
        logger.info("Vector store reset.")

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        data = self.collection.get(include=["metadatas"])
        return sorted({m.get("source", "") for m in data.get("metadatas") or [] if m})

    def search(
        self,
        query_embedding: Sequence[float],
        k: int = 4,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Return the k closest chunks as dicts:
        {"id", "text", "metadata", "score"} where score is cosine similarity (higher = closer).
        """
        total = self.count()
        if total == 0:
            return []
        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for id_, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"id": id_, "text": text, "metadata": meta or {}, "score": 1.0 - float(dist)})
        return hits


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str, int, float and bool metadata values (no None)."""
    clean = {}
    for key, value in meta.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean
