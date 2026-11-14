"""
Shows which PDF and page each answer came from.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from utils.helpers import escape_markdown, format_source, truncate


def render_sources(sources: list[dict[str, Any]], preview_chars: int = 500) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            score = src.get("score")
            line = f"**{i}. {escape_markdown(format_source(src))}**"
            if score is not None:
                line += f"  ·  relevance {score:.2f}"
            if "rerank_score" in src:
                line += f"  ·  rerank {src['rerank_score']:.0f}/10"
            st.markdown(line)
            st.caption(escape_markdown(truncate(src.get("text", ""), preview_chars)))
            source_path = src.get("metadata", {}).get("source")
            if source_path:
                st.caption(f"File: `{source_path}`")
            if i < len(sources):
                st.divider()
