"""
Test doubles so tests run fast and without Ollama.
"""
from __future__ import annotations

import math
import re

_VOCAB = [
    "refund", "policy", "days", "return", "warranty", "battery", "charge", "hours",
    "salary", "leave", "holiday", "employee", "python", "code", "install", "error",
]


class FakeEmbedder:
    """Bag-of-words vectors over a tiny vocabulary — similar words => similar vectors."""

    model = "fake-embed"

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        v = [float(sum(1 for w in words if w.startswith(term))) for term in _VOCAB]
        v.append(0.01)  # avoid an all-zero vector
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM:
    """Returns canned text and records the messages it received."""

    model = "fake-llm"

    def __init__(self, reply: str = "Fake answer [doc.pdf, p. 1]") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, temperature=None) -> str:
        self.calls.append(messages)
        return self.reply

    def stream_chat(self, messages, temperature=None):
        self.calls.append(messages)
        for word in self.reply.split(" "):
            yield word + " "

    def generate(self, prompt, system=None, temperature=None) -> str:
        return self.chat([{"role": "user", "content": prompt}], temperature)
