"""
Streamlit entry point. Run with:  streamlit run app.py

This file only puts the screen together and connects the pieces; all real
work happens in the other packages.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄", layout="wide")

from chains.rag_chain import RAGChain  # noqa: E402
from config import settings  # noqa: E402
from embeddings.ollama_embedder import OllamaEmbedder  # noqa: E402
from llm.ollama_llm import OllamaLLM, is_embedding_model, list_models  # noqa: E402
from memory.chat_history import ChatHistory  # noqa: E402
from retrieval.reranker import LLMReranker  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402
from ui.chat_view import handle_user_input, render_history  # noqa: E402
from ui.sidebar import render_sidebar, run_indexing  # noqa: E402
from utils.file_tracker import load_manifest  # noqa: E402
from utils.helpers import OllamaError  # noqa: E402
from vectorstore.vector_db import VectorDB  # noqa: E402


# ---------------------------------------------------------------- cached objects
@st.cache_resource(show_spinner=False)
def get_db() -> VectorDB:
    return VectorDB()


@st.cache_resource(show_spinner=False)
def get_embedder(model: str) -> OllamaEmbedder:
    return OllamaEmbedder(model=model)


@st.cache_data(ttl=30, show_spinner=False)
def get_models() -> tuple[list[str], bool]:
    try:
        return list_models(), True
    except OllamaError:
        return [], False


def index_stats(db: VectorDB) -> dict:
    manifest = load_manifest(settings.MANIFEST_PATH)
    return {
        "files": len(manifest["files"]),
        "chunks": db.count(),
        "embed_model": manifest.get("embed_model"),
        "sources": sorted(manifest["files"]),
    }


# ----------------------------------------------------------------------- page
def main() -> None:
    if "history" not in st.session_state:
        st.session_state.history = ChatHistory()
    history: ChatHistory = st.session_state.history

    db = get_db()
    models, ollama_ok = get_models()
    chat_models = [m for m in models if not is_embedding_model(m)] or models
    embed_models = [m for m in models if is_embedding_model(m)] or models
    stats = index_stats(db)

    state = render_sidebar(chat_models, embed_models, stats, ollama_ok)

    if state.clear_clicked:
        history.clear()
        st.rerun()

    if state.index_clicked:
        st.session_state.index_message = run_indexing(
            state.pdf_dir, state.rebuild, get_embedder(state.embed_model), db
        )
        st.rerun()

    st.title("📄 Chat with your PDFs")
    st.caption(f"Local RAG · chat model `{state.llm_model}` · everything runs on your machine via Ollama")

    if not ollama_ok:
        st.error("Ollama is not reachable. Start it with `ollama serve`, then refresh this page.")
        st.stop()

    chain = None
    if stats["chunks"] == 0:
        st.info("No documents indexed yet. Put PDFs in the folder shown in the sidebar and click **Index / Re-index PDFs**.")
    else:
        # Questions must use the same embedding model that built the index.
        query_embedder = get_embedder(stats["embed_model"] or state.embed_model)
        llm = OllamaLLM(model=state.llm_model, temperature=state.temperature)
        reranker = LLMReranker(llm) if state.use_reranker else None
        retriever = Retriever(query_embedder, db, reranker=reranker)
        chain = RAGChain(
            retriever,
            llm,
            top_k=state.top_k,
            score_threshold=state.score_threshold,
            source_filter=state.source_filter,
        )

    render_history(history)
    handle_user_input(chain, history)


main()
