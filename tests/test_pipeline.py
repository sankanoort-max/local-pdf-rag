"""End-to-end indexing test with a fake embedder: incremental updates must work."""
from ingestion.ingest_pipeline import run_ingestion
from tests.fakes import FakeEmbedder
from tests.test_loader import make_pdf
from utils.file_tracker import load_manifest
from vectorstore.vector_db import VectorDB


def test_incremental_indexing(tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    make_pdf(pdfs / "a.pdf", ["The refund policy allows returns within thirty days of purchase."])
    make_pdf(pdfs / "b.pdf", ["The battery needs three hours to charge completely from empty."])
    (pdfs / "broken.pdf").write_bytes(b"not a pdf")

    db = VectorDB(path=tmp_path / "vs", collection_name="pipeline_test")
    manifest = tmp_path / "vs" / "manifest.json"
    run = lambda **kw: run_ingestion(pdf_dir=pdfs, embedder=FakeEmbedder(), db=db, manifest_path=manifest, **kw)

    r1 = run()
    assert sorted(r1.new_files) == ["a.pdf", "b.pdf", "broken.pdf"]
    assert [f for f, _ in r1.errors] == ["broken.pdf"]
    assert db.count() == 2
    assert load_manifest(manifest)["embed_model"] == "fake-embed"

    r2 = run()  # nothing changed except the broken file is retried
    assert r2.new_files == ["broken.pdf"] and r2.chunks_added == 0

    (pdfs / "b.pdf").unlink()
    make_pdf(pdfs / "a.pdf", ["Updated refund policy: returns are accepted within sixty days now."])
    r3 = run()
    assert r3.changed_files == ["a.pdf"] and r3.deleted_files == ["b.pdf"]
    assert db.list_sources() == ["a.pdf"]

    r4 = run(rebuild=True)
    assert r4.rebuilt and db.count() == 1
