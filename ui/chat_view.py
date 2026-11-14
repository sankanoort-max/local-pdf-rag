"""
The main chat area: previous messages, the input box and streamed answers.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from chains.rag_chain import RAGChain
from memory.chat_history import ChatHistory
from ui.source_view import render_sources
from utils.helpers import OllamaError
from utils.logger import get_logger

logger = get_logger(__name__)


def render_history(history: ChatHistory) -> None:
    """Re-draw all previous messages (Streamlit redraws the page on every interaction)."""
    for message in history.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))


def handle_user_input(chain: Optional[RAGChain], history: ChatHistory, disabled: bool = False) -> None:
    """Show the input box; when the user asks something, answer it and save both messages."""
    question = st.chat_input("Ask a question about your PDFs…", disabled=disabled or chain is None)
    if not question or chain is None:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents…"):
                result = chain.stream(question, history)
            answer = st.write_stream(result.tokens)
        except OllamaError as exc:
            logger.error("Answering failed: %s", exc)
            st.error(str(exc))
            return
        if not isinstance(answer, str):
            answer = "".join(str(part) for part in answer)
        if result.question_used and result.question_used != question:
            st.caption(f"Searched for: _{result.question_used}_")
        render_sources(result.sources)

    history.add_user(question)
    history.add_assistant(answer, result.sources)
