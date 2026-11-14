"""
Finds the chunks that best match a question.
"""
from __future__ import annotations

from typing import Any, Optional

from config import settings
from embeddings.ollama_embedder import OllamaEmbedder
from retrieval.reranker import LLMReranker
from utils.helpers import Timer
from utils.logger import get_logger
from vectorstore.vector_db import VectorDB

logger = get_logger(__name__)


class Retriever:
    def __init__(self, embedder: OllamaEmbedder, db: VectorDB, reranker: Optional[LLMReranker] = None) -> None:
        self.embedder = embedder
        self.db = db
        self.reranker = reranker

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        sources: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Return up to `top_k` chunks: {"id", "text", "metadata", "score"}.

        sources: optional list of PDF source IDs to search in (empty = all files).
        """
        top_k = top_k or settings.TOP_K
        threshold = settings.SCORE_THRESHOLD if score_threshold is None else score_threshold
        fetch_k = max(top_k, settings.FETCH_K) if self.reranker else top_k
        where = {"source": {"$in": list(sources)}} if sources else None

        with Timer() as t:
            query_vec = self.embedder.embed_query(question)
            hits = self.db.search(query_vec, k=fetch_k, where=where)
        hits = [h for h in hits if h["score"] >= threshold]

        if self.reranker and len(hits) > 1:
            hits = self.reranker.rerank(question, hits, top_k)
        hits = hits[:top_k]

        logger.info(
            "Retrieved %d chunk(s) in %.2fs (scores: %s)",
            len(hits), t.elapsed, ", ".join(f"{h['score']:.2f}" for h in hits),
        )
        return hits
