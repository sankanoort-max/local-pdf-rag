"""
Reads PDF files page by page.

Every page keeps labels (metadata) — source, file name, page number, file
fingerprint — which travel with the text all the way to the answer so the UI
can show where information came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # PyMuPDF >= 1.24 can be imported as "pymupdf"; older versions only as "fitz"
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf

from utils.file_tracker import file_hash, find_pdfs, source_id
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoadError(Exception):
    """The file could not be opened as a PDF (corrupt, encrypted, ...)."""


@dataclass
class PageText:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def list_pdfs(pdf_dir: Path) -> list[Path]:
    """All PDFs in the folder (recursive)."""
    return find_pdfs(pdf_dir)


def load_pdf(path: Path, source: str | None = None, fingerprint: str | None = None) -> list[PageText]:
    """
    Extract text from every page of one PDF.

    Pages with no extractable text (e.g. scanned images) are skipped and logged;
    OCR could be added here later.
    """
    path = Path(path)
    source = source or path.name
    fingerprint = fingerprint or file_hash(path)

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise PDFLoadError(f"Cannot open {path.name}: {exc}") from exc

    with doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise PDFLoadError(f"{path.name} is password-protected")

        pages: list[PageText] = []
        empty_pages: list[int] = []
        for index, page in enumerate(doc):
            page_no = index + 1
            try:
                text = page.get_text("text") or ""
            except Exception as exc:
                logger.warning("%s page %d: text extraction failed (%s)", path.name, page_no, exc)
                continue
            if not text.strip():
                empty_pages.append(page_no)
                continue
            pages.append(
                PageText(
                    text=text,
                    metadata={
                        "source": source,
                        "file_name": path.name,
                        "page": page_no,
                        "total_pages": doc.page_count,
                        "file_hash": fingerprint,
                    },
                )
            )

    if empty_pages:
        logger.info(
            "%s: %d page(s) without text (scanned images?): %s",
            path.name, len(empty_pages), empty_pages[:20],
        )
    return pages


def load_folder(pdf_dir: Path) -> list[PageText]:
    """Convenience: load every PDF in a folder, skipping files that fail."""
    pages: list[PageText] = []
    for path in list_pdfs(pdf_dir):
        try:
            pages.extend(load_pdf(path, source=source_id(path, pdf_dir)))
        except PDFLoadError as exc:
            logger.warning("Skipping file: %s", exc)
    return pages
