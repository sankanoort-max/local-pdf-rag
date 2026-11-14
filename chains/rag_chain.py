"""
Journey B — answering a question.

question -> (rewrite follow-up) -> retrieve chunks -> build prompt -> LLM -> answer + sources

The UI only talks to this module, never directly to the retriever or the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from config import settings
from llm.ollama_llm import Message, OllamaLLM
from llm.prompts import SYSTEM_PROMPT, build_rag_prompt
from memory.chat_history import ChatHistory, condense_question
from retrieval.retriever import Retriever
from utils.logger import get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = (
    "I couldn't find anything relevant to that in your indexed PDFs. "
    "Try rephrasing the question, lowering the relevance threshold, or checking that the right files are indexed."
)


@dataclass
class RAGResult:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    question_used: str = ""


@dataclass
class RAGStream:
    tokens: Iterator[str]
    sources: list[dict[str, Any]] = field(default_factory=list)
    question_used: str = ""


class RAGChain:
    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaLLM,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        source_filter: Optional[list[str]] = None,
        max_history_turns: Optional[int] = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k or settings.TOP_K
        self.score_threshold = settings.SCORE_THRESHOLD if score_threshold is None else score_threshold
        self.source_filter = source_filter or None
        self.max_history_turns = settings.MAX_HISTORY_TURNS if max_history_turns is None else max_history_turns

    def _prepare(
        self, question: str, history: Optional[ChatHistory]
    ) -> tuple[str, list[dict[str, Any]], Optional[list[Message]]]:
        history = history or ChatHistory()
        search_question = condense_question(question, history, self.llm)
        chunks = self.retriever.retrieve(
            search_question,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            sources=self.source_filter,
        )
        if not chunks:
            return search_question, [], None

        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_rag_prompt(question, chunks, history.as_text(self.max_history_turns)),
            },
        ]
        return search_question, chunks, messages

    def ask(self, question: str, history: Optional[ChatHistory] = None) -> RAGResult:
        """Full answer in one go (CLI, tests)."""
        used, chunks, messages = self._prepare(question, history)
        if messages is None:
            return RAGResult(answer=NOT_FOUND_MESSAGE, sources=[], question_used=used)
        answer = self.llm.chat(messages)
        return RAGResult(answer=answer, sources=chunks, question_used=used)

    def stream(self, question: str, history: Optional[ChatHistory] = None) -> RAGStream:
        """Answer streamed token by token (UI). Sources are known before streaming starts."""
        used, chunks, messages = self._prepare(question, history)
        if messages is None:
            return RAGStream(tokens=iter([NOT_FOUND_MESSAGE]), sources=[], question_used=used)
        return RAGStream(tokens=self.llm.stream_chat(messages), sources=chunks, question_used=used)
