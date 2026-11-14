"""
The settings panel on the left: folder, models, retrieval settings, index button.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from config import settings
from embeddings.ollama_embedder import OllamaEmbedder
from ingestion.ingest_pipeline import IngestionReport, run_ingestion
from vectorstore.vector_db import VectorDB


@dataclass
class SidebarState:
    pdf_dir: str
    llm_model: str
    embed_model: str
    top_k: int
    temperature: float
    score_threshold: float
    use_reranker: bool
    source_filter: list[str] = field(default_factory=list)
    index_clicked: bool = False
    rebuild: bool = False
    clear_clicked: bool = False


def _with_default(options: list[str], default: str) -> list[str]:
    """Make sure the configured default model is selectable even if not listed."""
    return options if default in options else [default, *options]


def render_sidebar(
    chat_models: list[str],
    embed_models: list[str],
    stats: dict[str, Any],
    ollama_ok: bool,
) -> SidebarState:
    sb = st.sidebar
    sb.title("⚙️ Settings")

    if ollama_ok:
        sb.success("Ollama is running", icon="✅")
    else:
        sb.error(f"Cannot reach Ollama at {settings.OLLAMA_BASE_URL}. Start it with `ollama serve`.", icon="⚠️")

    # ------------------------------------------------------------ documents
    sb.subheader("📂 Documents")
    pdf_dir = sb.text_input("PDF folder", value=str(settings.PDF_DIR))
    sb.caption(f"Indexed: **{stats['files']}** file(s) · **{stats['chunks']}** chunk(s)")
    if stats.get("embed_model"):
        sb.caption(f"Index built with `{stats['embed_model']}`")

    rebuild = sb.checkbox("Rebuild from scratch", value=False, help="Delete the index and re-process every PDF.")
    index_clicked = sb.button("📥 Index / Re-index PDFs", disabled=not ollama_ok, type="primary")

    message: Optional[tuple[str, str]] = st.session_state.get("index_message")
    if message:
        kind, text = message
        getattr(sb, kind)(text)

    # --------------------------------------------------------------- models
    sb.subheader("🤖 Models")
    llm_options = _with_default(chat_models, settings.LLM_MODEL)
    llm_model = sb.selectbox("Chat model", llm_options, index=llm_options.index(settings.LLM_MODEL))

    default_embed = stats.get("embed_model") or settings.EMBED_MODEL
    embed_options = _with_default(embed_models, default_embed)
    embed_model = sb.selectbox(
        "Embedding model",
        embed_options,
        index=embed_options.index(default_embed),
        help="Used when indexing. Questions always use the model that built the index.",
    )
    if stats.get("embed_model") and embed_model != stats["embed_model"]:
        sb.warning(
            f"Questions keep using `{stats['embed_model']}` until you re-index. "
            "Re-indexing with a new embedding model rebuilds the whole index."
        )

    # ------------------------------------------------------------ retrieval
    sb.subheader("🔎 Retrieval")
    top_k = sb.slider("Chunks per answer (top-k)", 1, 15, settings.TOP_K)
    score_threshold = sb.slider(
        "Minimum relevance", 0.0, 1.0, float(settings.SCORE_THRESHOLD), 0.05,
        help="Chunks less similar than this are ignored.",
    )
    temperature = sb.slider("Temperature", 0.0, 1.0, float(settings.TEMPERATURE), 0.05)
    use_reranker = sb.toggle(
        "Rerank with LLM (slower)", value=settings.USE_RERANKER,
        help=f"Fetch {settings.FETCH_K} chunks and let the chat model pick the best.",
    )
    source_filter = sb.multiselect(
        "Search only in (optional)", stats.get("sources", []), help="Leave empty to search all files."
    )

    sb.divider()
    clear_clicked = sb.button("🗑️ Clear chat")

    return SidebarState(
        pdf_dir=pdf_dir,
        llm_model=llm_model,
        embed_model=embed_model,
        top_k=top_k,
        temperature=temperature,
        score_threshold=score_threshold,
        use_reranker=use_reranker,
        source_filter=source_filter,
        index_clicked=index_clicked,
        rebuild=rebuild,
        clear_clicked=clear_clicked,
    )


def run_indexing(pdf_dir: str, rebuild: bool, embedder: OllamaEmbedder, db: VectorDB) -> tuple[str, str]:
    """
    Run the ingestion pipeline with a progress bar in the sidebar.
    Returns (message_kind, message_text) for st.sidebar.<kind>(text).
    """
    bar = st.sidebar.progress(0.0, text="Scanning folder…")

    def on_progress(done: int, total: int, msg: str) -> None:
        bar.progress(done / total if total else 1.0, text=msg)

    try:
        report: IngestionReport = run_ingestion(
            pdf_dir=Path(pdf_dir), rebuild=rebuild, embedder=embedder, db=db, progress_cb=on_progress
        )
    except Exception as exc:  # FileNotFoundError, OllamaError, ...
        bar.empty()
        return "error", f"Indexing failed: {exc}"

    bar.empty()
    kind = "warning" if report.errors or report.empty_files else "success"
    return kind, "Indexing complete.\n\n" + report.summary().replace("\n", "\n\n")
