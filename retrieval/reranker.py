"""
Optional second sorting step.

The retriever quickly fetches more candidates than needed (FETCH_K); the
reranker asks the LLM to score each one and keeps the best. It improves
quality but costs one LLM call per candidate, so it is off by default
(USE_RERANKER=false).
"""
from __future__ import annotations

import re
from typing import Any

from llm.ollama_llm import OllamaLLM
from llm.prompts import RERANK_TEMPLATE
from utils.helpers import OllamaError
from utils.logger import get_logger

logger = get_logger(__name__)

_NUMBER = re.compile(r"\d+(\.\d+)?")


class LLMReranker:
    def __init__(self, llm: OllamaLLM, max_passage_chars: int = 1500) -> None:
        self.llm = llm
        self.max_passage_chars = max_passage_chars

    def _score(self, question: str, passage: str) -> float:
        prompt = RERANK_TEMPLATE.format(question=question, passage=passage[: self.max_passage_chars])
        reply = self.llm.generate(prompt, temperature=0.0)
        match = _NUMBER.search(reply)
        return min(10.0, float(match.group())) if match else 0.0

    def rerank(self, question: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        scored = []
        for chunk in chunks:
            try:
                score = self._score(question, chunk["text"])
            except OllamaError as exc:
                logger.warning("Reranking failed, keeping vector order: %s", exc)
                return chunks[:top_k]
            scored.append({**chunk, "rerank_score": score})
        # sort by LLM score, break ties with the original vector similarity
        scored.sort(key=lambda c: (c["rerank_score"], c.get("score", 0.0)), reverse=True)
        return scored[:top_k]
