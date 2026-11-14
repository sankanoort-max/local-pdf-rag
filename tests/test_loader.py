from pathlib import Path

import pytest

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf

from ingestion.pdf_loader import PDFLoadError, load_pdf
from ingestion.text_cleaner import clean_pages, clean_text


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_load_pdf_extracts_text_and_page_numbers(tmp_path):
    pdf = make_pdf(tmp_path / "sample.pdf", ["Hello first page", "Second page here", "Third page"])
    pages = load_pdf(pdf, source="sample.pdf")

    assert [p.metadata["page"] for p in pages] == [1, 2, 3]
    assert "Hello first page" in pages[0].text
    assert pages[1].metadata["file_name"] == "sample.pdf"
    assert pages[0].metadata["file_hash"]


def test_empty_pages_are_skipped(tmp_path):
    pdf = make_pdf(tmp_path / "gaps.pdf", ["Text", "", "More text"])
    pages = load_pdf(pdf)
    assert [p.metadata["page"] for p in pages] == [1, 3]


def test_broken_file_raises_clear_error(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(PDFLoadError):
        load_pdf(bad)


def test_clean_text_joins_hyphens_and_lines():
    raw = "This is an informa-\ntion line\nthat wraps.\n\nNew paragraph.\n  12  \n"
    assert clean_text(raw) == "This is an information line that wraps.\n\nNew paragraph."


def test_repeated_headers_are_removed(tmp_path):
    topics = ["revenue grew", "costs fell", "hiring paused", "new office", "outlook stable"]
    texts = [f"ACME Corp Annual Report\nIn 2023 {t} by 12 percent.\nConfidential - page {i} of 5"
             for i, t in enumerate(topics, start=1)]
    pdf = make_pdf(tmp_path / "report.pdf", texts)
    pages = clean_pages(load_pdf(pdf))
    assert len(pages) == 5
    assert all("ACME Corp" not in p.text and "Confidential" not in p.text for p in pages)
    assert pages[2].text == "In 2023 hiring paused by 12 percent."  # body lines with numbers are kept
