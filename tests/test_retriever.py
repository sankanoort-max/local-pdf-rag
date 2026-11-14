import pytest

from ingestion.chunker import Chunk
from retrieval.retriever import Retriever
from tests.fakes import FakeEmbedder
from vectorstore.vector_db import VectorDB

DOCS = [
    ("policy.pdf", 1, "Our refund policy allows a return within 30 days of purchase."),
    ("manual.pdf", 4, "The battery needs 3 hours to fully charge."),
    ("hr.pdf", 2, "Each employee gets 20 days of paid leave and 10 holidays."),
    ("guide.pdf", 7, "To install the python package, run the install command. Check the error log."),
]


@pytest.fixture
def retriever(tmp_path):
    db = VectorDB(path=tmp_path / "vs", collection_name="test")
    embedder = FakeEmbedder()
    chunks = [
        Chunk(id=f"{src}-{page}", text=text, metadata={"source": src, "file_name": src, "page": page})
        for src, page, text in DOCS
    ]
    db.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    return Retriever(embedder, db)


def test_best_match_comes_first(retriever):
    hits = retriever.retrieve("How many hours to charge the battery?", top_k=2, score_threshold=0.0)
    assert hits[0]["metadata"]["file_name"] == "manual.pdf"
    assert hits[0]["metadata"]["page"] == 4
    assert hits[0]["score"] >= hits[1]["score"]


def test_threshold_removes_unrelated(retriever):
    hits = retriever.retrieve("What is the refund policy?", top_k=4, score_threshold=0.5)
    assert [h["metadata"]["source"] for h in hits] == ["policy.pdf"]


def test_source_filter(retriever):
    hits = retriever.retrieve("refund policy", top_k=4, score_threshold=0.0, sources=["hr.pdf"])
    assert hits and all(h["metadata"]["source"] == "hr.pdf" for h in hits)


def test_delete_and_count(retriever):
    db = retriever.db
    assert db.count() == 4
    db.delete_source("policy.pdf")
    assert db.count() == 3
    assert "policy.pdf" not in db.list_sources()
    db.reset()
    assert db.count() == 0
    assert retriever.retrieve("anything", score_threshold=0.0) == []
