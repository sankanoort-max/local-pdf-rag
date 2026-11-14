"""
Index PDFs from the command line (no UI needed).

    python scripts/build_index.py
    python scripts/build_index.py --pdf-dir "D:/my pdfs" --rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from embeddings.ollama_embedder import OllamaEmbedder  # noqa: E402
from ingestion.ingest_pipeline import run_ingestion  # noqa: E402
from utils.helpers import OllamaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index PDFs into the local vector store.")
    parser.add_argument("--pdf-dir", default=str(settings.PDF_DIR), help="Folder with PDF files")
    parser.add_argument("--rebuild", action="store_true", help="Delete the index and re-process everything")
    parser.add_argument("--embed-model", default=settings.EMBED_MODEL, help="Ollama embedding model")
    args = parser.parse_args()

    def progress(done: int, total: int, msg: str) -> None:
        print(f"[{done}/{total}] {msg}", flush=True)

    try:
        report = run_ingestion(
            pdf_dir=Path(args.pdf_dir),
            rebuild=args.rebuild,
            embedder=OllamaEmbedder(model=args.embed_model),
            progress_cb=progress,
        )
    except (OllamaError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\n" + report.summary())
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
