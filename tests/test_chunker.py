import pytest

from ingestion.chunker import chunk_pages, split_text
from ingestion.pdf_loader import PageText

LONG_TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(300))


def test_chunks_respect_max_size():
    chunks = split_text(LONG_TEXT, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_no_text_is_lost():
    chunks = split_text(LONG_TEXT, chunk_size=300, overlap=50)
    for i in range(300):
        assert any(f"Sentence number {i} " in c for c in chunks), f"sentence {i} missing"


def test_neighbouring_chunks_overlap():
    chunks = split_text(LONG_TEXT, chunk_size=300, overlap=80)
    for a, b in zip(chunks, chunks[1:]):
        tail_words = a.split()[-3:]
        assert " ".join(tail_words) in b


def test_short_text_is_one_chunk():
    assert split_text("Just a short page.", chunk_size=500, overlap=50) == ["Just a short page."]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_metadata_and_ids_are_attached():
    pages = [
        PageText(LONG_TEXT, {"source": "a.pdf", "file_name": "a.pdf", "page": 1, "file_hash": "abc123"}),
        PageText("Short page two with enough characters to keep.", {"source": "a.pdf", "file_name": "a.pdf", "page": 2, "file_hash": "abc123"}),
    ]
    chunks = chunk_pages(pages, chunk_size=400, overlap=50, min_chars=10)
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    assert all(c.metadata["source"] == "a.pdf" for c in chunks)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0].startswith("abc123-p1-c0")
