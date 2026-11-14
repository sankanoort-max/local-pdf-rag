"""
Tidies up raw PDF text before chunking.

- removes headers/footers that repeat on most pages
- removes standalone page-number lines
- re-joins words split by a hyphen at a line break
- joins wrapped lines into paragraphs and normalises whitespace
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from ingestion.pdf_loader import PageText

_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINES = re.compile(r"\n\s*\n+")
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_SPACES = re.compile(r"[ \t ]+")


def _normalise_line(line: str, page_no: int | None = None) -> str:
    """
    Key used to spot repeated headers/footers. Only the page's own number is
    masked, so 'Page 3 of 10' on page 3 and 'Page 4 of 10' on page 4 match,
    while body lines that merely contain other numbers stay distinct.
    """
    key = " ".join(line.lower().split())
    if page_no is not None:
        key = re.sub(rf"(?<!\d){page_no}(?!\d)", "#", key, count=1)
    return key


def find_repeated_lines(
    pages: list[tuple[str, int | None]], min_ratio: float = 0.6, max_len: int = 100
) -> set[str]:
    """Short lines that appear on at least `min_ratio` of the pages. `pages` = [(text, page_no)]."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for text, page_no in pages:
        seen = {_normalise_line(l, page_no) for l in text.splitlines() if 0 < len(l.strip()) <= max_len}
        counts.update(seen)
    threshold = max(2, int(len(pages) * min_ratio))
    return {line for line, n in counts.items() if n >= threshold}


def clean_text(text: str, drop_lines: set[str] | None = None, page_no: int | None = None) -> str:
    """Clean the text of a single page."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")  # soft hyphen

    kept = []
    for line in text.split("\n"):
        if _PAGE_NUMBER_LINE.match(line):
            continue
        if drop_lines and line.strip() and _normalise_line(line, page_no) in drop_lines:
            continue
        kept.append(line.rstrip())
    text = "\n".join(kept)

    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # "informa-\ntion" -> "information"
    text = _MULTI_NEWLINES.sub("\n\n", text)          # keep paragraph breaks
    text = _SINGLE_NEWLINE.sub(" ", text)             # join wrapped lines
    text = _SPACES.sub(" ", text)
    text = "\n\n".join(p.strip() for p in text.split("\n\n"))
    return text.strip()


def clean_pages(pages: list[PageText]) -> list[PageText]:
    """
    Clean all pages of ONE document. Header/footer detection needs to see the
    whole document, so call this per file, not on a mix of files.
    """
    repeated = find_repeated_lines([(p.text, p.metadata.get("page")) for p in pages])
    cleaned = []
    for page in pages:
        text = clean_text(page.text, drop_lines=repeated, page_no=page.metadata.get("page"))
        if text:
            cleaned.append(PageText(text=text, metadata=dict(page.metadata)))
    return cleaned
