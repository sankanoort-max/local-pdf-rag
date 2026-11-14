"""
Keeps track of which PDFs are already indexed.

Each PDF gets a fingerprint (SHA-256 hash of its bytes). A small JSON
"manifest" stored next to the vector database remembers the fingerprint of
every indexed file, plus which embedding model built the index. Comparing the
folder with the manifest tells us which files are new, changed or deleted, so
re-indexing only touches what actually changed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_VERSION = 1


@dataclass
class FileChanges:
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.new or self.changed or self.deleted)


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
def find_pdfs(pdf_dir: Path) -> list[Path]:
    """All PDF files in the folder and its sub-folders, sorted by path."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        return []
    return sorted(p for p in pdf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def source_id(path: Path, pdf_dir: Path) -> str:
    """Stable ID for a file: its path relative to the PDF folder, with '/' separators."""
    return Path(path).resolve().relative_to(Path(pdf_dir).resolve()).as_posix()


def file_hash(path: Path, block_size: int = 1 << 20) -> str:
    """SHA-256 fingerprint of a file, read in blocks so big PDFs are fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_pdfs(pdf_dir: Path) -> dict[str, str]:
    """Map of source_id -> fingerprint for every PDF currently in the folder."""
    result: dict[str, str] = {}
    for path in find_pdfs(pdf_dir):
        try:
            result[source_id(path, pdf_dir)] = file_hash(path)
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
    return result


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def empty_manifest() -> dict[str, Any]:
    return {"version": MANIFEST_VERSION, "embed_model": None, "pdf_dir": None, "files": {}}


def load_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return empty_manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data.get("files"), dict):
            raise ValueError("missing 'files'")
        return {**empty_manifest(), **data}
    except (ValueError, OSError) as exc:
        logger.warning("Manifest %s is unreadable (%s); starting fresh.", path, exc)
        return empty_manifest()


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic on the same filesystem


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare(current: dict[str, str], indexed_files: dict[str, dict[str, Any]]) -> FileChanges:
    """Compare the folder scan with the manifest's 'files' section."""
    changes = FileChanges()
    for src, digest in current.items():
        entry = indexed_files.get(src)
        if entry is None:
            changes.new.append(src)
        elif entry.get("hash") != digest:
            changes.changed.append(src)
        else:
            changes.unchanged.append(src)
    changes.deleted = sorted(set(indexed_files) - set(current))
    return changes
