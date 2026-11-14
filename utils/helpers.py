"""
Small helpers shared by several modules: source formatting, text previews,
timing, and a retry wrapper for calls to the local Ollama server.
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or a model is missing."""


def call_with_retry(fn: Callable[[], T], *, what: str, attempts: int = 3, base_delay: float = 1.0) -> T:
    """
    Call an Ollama function with retries and exponential back-off.

    A missing model (HTTP 404) is not retried because waiting won't fix it.
    Any final failure is converted into an OllamaError with a friendly message.
    """
    import ollama  # local import keeps this module lightweight

    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise OllamaError(
                    f"Ollama could not find the model while {what}: {exc.error}. "
                    "Pull it first, e.g. `ollama pull <model-name>`."
                ) from exc
            last_exc = exc
        except Exception as exc:  # connection refused, timeouts, ...
            last_exc = exc
        if attempt < attempts:
            time.sleep(base_delay * (2 ** (attempt - 1)))

    raise OllamaError(
        f"Could not reach Ollama while {what}: {last_exc}. "
        "Make sure Ollama is running (`ollama serve`) and OLLAMA_BASE_URL is correct."
    ) from last_exc


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def format_source(item: dict[str, Any]) -> str:
    """'report.pdf — page 4' from a chunk dict or its metadata."""
    meta = item.get("metadata", item)
    name = meta.get("file_name") or meta.get("source", "unknown file")
    page = meta.get("page")
    return f"{name} — page {page}" if page is not None else str(name)


def truncate(text: str, max_chars: int = 300) -> str:
    """Shorten text for previews without cutting in the middle of a word."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " …"


def escape_markdown(text: str) -> str:
    """Escape characters that Streamlit markdown would otherwise interpret."""
    for ch in ("\\", "$", "*", "_", "`", "#", ">", "|", "[", "]"):
        text = text.replace(ch, "\\" + ch)
    return text


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
class Timer:
    """
    Context manager that measures elapsed seconds.

        with Timer() as t:
            do_work()
        print(t.elapsed)
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start
