"""
All instructions given to the model live here, so they are easy to tune.
"""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the context \
taken from the user's PDF documents.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- If the context does not contain the answer, say: "I don't know based on the provided documents."
- Cite the sources you used in square brackets, like [report.pdf, p. 3].
- Be concise and clear. Use bullet points for lists or steps.
- If sources disagree, say so and cite both."""

RAG_TEMPLATE = """Conversation so far:
{history}

Context from the documents:
{context}

Question: {question}

Answer (with citations):"""

CONDENSE_TEMPLATE = """Given the conversation below and a follow-up question, rewrite the \
follow-up as a single standalone question that can be understood without the conversation. \
Keep names, numbers and key terms. If it is already standalone, return it unchanged. \
Reply with the question only.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""

RERANK_TEMPLATE = """Rate how useful the passage is for answering the question, on a scale \
from 0 (not relevant) to 10 (directly answers it). Reply with a single number only.

Question: {question}

Passage:
{passage}

Score:"""


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Show each chunk to the model with a label it can cite."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        label = f"[{meta.get('file_name', meta.get('source', 'unknown'))}, p. {meta.get('page', '?')}]"
        blocks.append(f"Source {i} {label}\n{chunk['text'].strip()}")
    return "\n\n---\n\n".join(blocks)


def build_rag_prompt(question: str, chunks: list[dict[str, Any]], history_text: str = "") -> str:
    return RAG_TEMPLATE.format(
        history=history_text or "(none)",
        context=format_context(chunks),
        question=question.strip(),
    )
