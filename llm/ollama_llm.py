"""
Talks to the local Ollama chat model.
"""
from __future__ import annotations

from typing import Any, Iterator, Optional

import ollama

from config import settings
from utils.helpers import OllamaError, call_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

_EMBED_HINTS = ("embed", "bge", "minilm", "gte-", "paraphrase")


class OllamaLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model or settings.LLM_MODEL
        self.temperature = settings.TEMPERATURE if temperature is None else temperature
        self.num_ctx = num_ctx or settings.NUM_CTX
        self._client = ollama.Client(host=base_url or settings.OLLAMA_BASE_URL, timeout=settings.OLLAMA_TIMEOUT)

    def _options(self, temperature: Optional[float]) -> dict[str, Any]:
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
        }

    def chat(self, messages: list[Message], temperature: Optional[float] = None) -> str:
        """Send messages and return the full answer text."""
        response = call_with_retry(
            lambda: self._client.chat(model=self.model, messages=messages, options=self._options(temperature)),
            what=f"generating with '{self.model}'",
        )
        return response["message"]["content"]

    def stream_chat(self, messages: list[Message], temperature: Optional[float] = None) -> Iterator[str]:
        """Yield the answer piece by piece as the model writes it."""
        try:
            stream = self._client.chat(
                model=self.model, messages=messages, options=self._options(temperature), stream=True
            )
            for part in stream:
                token = part["message"]["content"]
                if token:
                    yield token
        except ollama.ResponseError as exc:
            raise OllamaError(f"Ollama error with model '{self.model}': {exc.error}") from exc
        except Exception as exc:
            raise OllamaError(
                f"Lost connection to Ollama while generating: {exc}. Is `ollama serve` running?"
            ) from exc

    def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """Convenience for one-off prompts (question rewriting, reranking)."""
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature)


# --------------------------------------------------------------------------- #
# Model discovery (used by the sidebar dropdowns)
# --------------------------------------------------------------------------- #
def list_models(base_url: Optional[str] = None) -> list[str]:
    """Names of models installed in Ollama (':latest' tag stripped)."""
    client = ollama.Client(host=base_url or settings.OLLAMA_BASE_URL, timeout=10)
    response = call_with_retry(client.list, what="listing models", attempts=1)
    names = []
    for m in response["models"]:
        name = getattr(m, "model", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(name[: -len(":latest")] if name.endswith(":latest") else name)
    return sorted(set(names))


def is_embedding_model(name: str) -> bool:
    """Best-effort guess based on common model names."""
    return any(hint in name.lower() for hint in _EMBED_HINTS)
