"""
Cuts page text into small, overlapping chunks.

Splitting tries natural boundaries first (paragraphs, then lines, then
sentences, then words) and only cuts mid-word as a last resort. Neighbouring
chunks overlap a little so a sentence on the edge is never lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import settings
from ingestion.pdf_loader import PageText

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_recursive(text: str, separators: list[str], size: int) -> list[str]:
    """Break text into pieces no longer than `size`, preferring early separators."""
    if len(text) <= size:
        return [text]

    sep = next((s for s in separators if s == "" or s in text), "")
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]  # keep the separator
    remaining = separators[separators.index(sep) + 1 :]

    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > size:
            out.extend(_split_recursive(piece, remaining, size))
        else:
            out.append(piece)
    return out


def split_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split one text into chunks of at most `chunk_size` characters with overlap."""
    size = chunk_size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    pieces = _split_recursive(text, _SEPARATORS, size)

    chunks: list[str] = []
    window: list[str] = []
    window_len = 0
    for piece in pieces:
        if window and window_len + len(piece) > size:
            chunks.append("".join(window).strip())
            # keep the tail of the window as overlap for the next chunk
            while window and (window_len > overlap or window_len + len(piece) > size):
                window_len -= len(window[0])
                window.pop(0)
        window.append(piece)
        window_len += len(piece)
    if window:
        chunks.append("".join(window).strip())

    return [c for c in chunks if c]


def chunk_pages(
    pages: list[PageText],
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_chars: int | None = None,
) -> list[Chunk]:
    """Turn cleaned pages into chunks that carry the page's labels and a unique ID."""
    min_chars = settings.MIN_CHUNK_CHARS if min_chars is None else min_chars
    chunks: list[Chunk] = []
    for page in pages:
        meta = page.metadata
        prefix = f"{str(meta.get('file_hash', 'nohash'))[:16]}-p{meta.get('page', 0)}"
        for i, text in enumerate(split_text(page.text, chunk_size, overlap)):
            if len(text) < min_chars:
                continue
            chunks.append(
                Chunk(id=f"{prefix}-c{i}", text=text, metadata={**meta, "chunk_index": i})
            )
    return chunks
