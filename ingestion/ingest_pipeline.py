"""
Journey A — indexing.

Runs: find changed files -> load -> clean -> chunk -> embed -> store, and keeps
the manifest up to date. Used by both the Streamlit "Index" button and
scripts/build_index.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from config import settings
from embeddings.ollama_embedder import OllamaEmbedder
from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_pdf
from ingestion.text_cleaner import clean_pages
from utils.file_tracker import compare, empty_manifest, load_manifest, save_manifest, scan_pdfs
from utils.helpers import OllamaError, Timer
from utils.logger import get_logger
from vectorstore.vector_db import VectorDB

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]  # (done, total, message)


@dataclass
class IngestionReport:
    new_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    unchanged_files: int = 0
    empty_files: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    chunks_added: int = 0
    rebuilt: bool = False
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"New: {len(self.new_files)} · Changed: {len(self.changed_files)} · "
            f"Removed: {len(self.deleted_files)} · Unchanged: {self.unchanged_files}",
            f"Chunks added: {self.chunks_added} · Time: {self.seconds:.1f}s",
        ]
        if self.rebuilt:
            lines.append("The index was rebuilt from scratch.")
        if self.empty_files:
            lines.append(f"No usable text (scanned images or very short pages?): {', '.join(self.empty_files)}")
        if self.errors:
            lines.append("Errors: " + "; ".join(f"{f}: {m}" for f, m in self.errors))
        return "\n".join(lines)


def _index_file(path: Path, source: str, fingerprint: str, embedder: OllamaEmbedder, db: VectorDB) -> int:
    """Index one PDF and return the number of chunks stored."""
    pages = load_pdf(path, source=source, fingerprint=fingerprint)
    pages = clean_pages(pages)
    chunks = chunk_pages(pages)
    if not chunks:
        return 0
    vectors = embedder.embed_documents([c.text for c in chunks])
    db.add_chunks(chunks, vectors)
    return len(chunks)


def run_ingestion(
    pdf_dir: Optional[Path] = None,
    rebuild: bool = False,
    embedder: Optional[OllamaEmbedder] = None,
    db: Optional[VectorDB] = None,
    progress_cb: Optional[ProgressCallback] = None,
    manifest_path: Optional[Path] = None,
) -> IngestionReport:
    """
    Bring the vector database in line with the PDF folder.

    Only new or changed files are processed; deleted files are removed. If the
    embedding model differs from the one that built the index, the index is
    rebuilt automatically (old and new vectors cannot be mixed).
    """
    pdf_dir = Path(pdf_dir or settings.PDF_DIR).expanduser().resolve()
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {pdf_dir}")

    embedder = embedder or OllamaEmbedder()
    db = db or VectorDB()
    manifest_path = Path(manifest_path or settings.MANIFEST_PATH)
    progress = progress_cb or (lambda done, total, msg: None)

    report = IngestionReport()
    manifest = load_manifest(manifest_path)

    previous_model = manifest.get("embed_model")
    if previous_model and previous_model != embedder.model and not rebuild:
        logger.warning("Embedding model changed (%s -> %s); rebuilding index.", previous_model, embedder.model)
        rebuild = True
    if db.count() > 0 and not manifest["files"] and not rebuild:
        logger.warning("Vector store has data but no manifest; rebuilding index to stay consistent.")
        rebuild = True
    if rebuild:
        db.reset()
        manifest = empty_manifest()
        report.rebuilt = True

    with Timer() as timer:
        current = scan_pdfs(pdf_dir)
        changes = compare(current, manifest["files"])
        report.new_files = changes.new
        report.changed_files = changes.changed
        report.deleted_files = changes.deleted
        report.unchanged_files = len(changes.unchanged)
        logger.info(
            "Scan of %s: %d new, %d changed, %d deleted, %d unchanged",
            pdf_dir, len(changes.new), len(changes.changed), len(changes.deleted), len(changes.unchanged),
        )

        for src in changes.deleted:
            db.delete_source(src)
            manifest["files"].pop(src, None)

        to_process = changes.new + changes.changed
        total = len(to_process)
        try:
            for i, src in enumerate(to_process):
                progress(i, total, f"Indexing {src} ({i + 1}/{total})")
                db.delete_source(src)  # remove the old version of a changed file
                try:
                    n = _index_file(pdf_dir / src, src, current[src], embedder, db)
                except OllamaError:
                    manifest["files"].pop(src, None)
                    raise
                except Exception as exc:  # one bad file must not stop the rest
                    logger.exception("Failed to index %s", src)
                    report.errors.append((src, str(exc)))
                    manifest["files"].pop(src, None)
                    continue

                manifest["files"][src] = {
                    "hash": current[src],
                    "chunks": n,
                    "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                report.chunks_added += n
                if n == 0:
                    report.empty_files.append(src)
                logger.info("Indexed %s: %d chunks", src, n)
        finally:
            # Save progress even if Ollama failed half-way, so finished files are not redone.
            manifest["embed_model"] = embedder.model
            manifest["pdf_dir"] = str(pdf_dir)
            save_manifest(manifest, manifest_path)

    report.seconds = timer.elapsed
    progress(total, total, "Done")
    logger.info("Ingestion finished. %s", report.summary().replace("\n", " | "))
    return report
