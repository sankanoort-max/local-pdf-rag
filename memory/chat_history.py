"""
Remembers the conversation so follow-up questions work.

Messages are stored as dicts: {"role": "user"|"assistant", "content": str, "sources": list}.
The "sources" key is only used by the UI; prompts only see role + content.
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from config import settings
from llm.prompts import CONDENSE_TEMPLATE
from utils.helpers import OllamaError
from utils.logger import get_logger

if TYPE_CHECKING:
    from llm.ollama_llm import OllamaLLM

logger = get_logger(__name__)


class ChatHistory:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, sources: Optional[list[dict[str, Any]]] = None) -> None:
        self.messages.append({"role": "assistant", "content": content, "sources": sources or []})

    def clear(self) -> None:
        self.messages.clear()

    def is_empty(self) -> bool:
        return not self.messages

    def recent(self, max_turns: Optional[int] = None) -> list[dict[str, Any]]:
        """The last N question/answer pairs."""
        turns = settings.MAX_HISTORY_TURNS if max_turns is None else max_turns
        return self.messages[-2 * turns :] if turns > 0 else []

    def as_text(self, max_turns: Optional[int] = None, max_chars_per_message: int = 800) -> str:
        """Recent history as plain text for prompts."""
        lines = []
        for m in self.recent(max_turns):
            who = "User" if m["role"] == "user" else "Assistant"
            content = m["content"].strip()
            if len(content) > max_chars_per_message:
                content = content[:max_chars_per_message] + " …"
            lines.append(f"{who}: {content}")
        return "\n".join(lines)


def condense_question(question: str, history: ChatHistory, llm: "OllamaLLM") -> str:
    """
    Rewrite a follow-up ("and the second one?") into a standalone question so
    the search finds the right chunks. Falls back to the original on error.
    """
    if history.is_empty():
        return question
    prompt = CONDENSE_TEMPLATE.format(history=history.as_text(), question=question)
    try:
        rewritten = llm.generate(prompt, temperature=0.0).strip().strip('"').strip()
    except OllamaError as exc:
        logger.warning("Could not rewrite question, using original: %s", exc)
        return question
    # Guard against the model answering instead of rewriting
    if not rewritten or len(rewritten) > 4 * len(question) + 200:
        return question
    logger.info("Condensed question: %r -> %r", question, rewritten)
    return rewritten
