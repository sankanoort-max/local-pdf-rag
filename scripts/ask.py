"""
Ask questions from the terminal — useful to test retrieval + answers before the UI.

    python scripts/ask.py "What is the refund policy?"
    python scripts/ask.py            # interactive mode, type 'exit' to quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chains.rag_chain import RAGChain  # noqa: E402
from config import settings  # noqa: E402
from embeddings.ollama_embedder import OllamaEmbedder  # noqa: E402
from llm.ollama_llm import OllamaLLM  # noqa: E402
from memory.chat_history import ChatHistory  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402
from utils.file_tracker import load_manifest  # noqa: E402
from utils.helpers import OllamaError, format_source  # noqa: E402
from vectorstore.vector_db import VectorDB  # noqa: E402


def build_chain(llm_model: str, top_k: int) -> RAGChain:
    manifest = load_manifest(settings.MANIFEST_PATH)
    embedder = OllamaEmbedder(model=manifest.get("embed_model") or settings.EMBED_MODEL)
    retriever = Retriever(embedder, VectorDB())
    return RAGChain(retriever, OllamaLLM(model=llm_model), top_k=top_k)


def answer(chain: RAGChain, question: str, history: ChatHistory) -> None:
    result = chain.ask(question, history)
    print("\n" + result.answer + "\n")
    for i, src in enumerate(result.sources, start=1):
        print(f"  [{i}] {format_source(src)}  (score {src['score']:.2f})")
    print()
    history.add_user(question)
    history.add_assistant(result.answer, result.sources)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about your indexed PDFs.")
    parser.add_argument("question", nargs="?", help="Question (omit for interactive mode)")
    parser.add_argument("--model", default=settings.LLM_MODEL, help="Ollama chat model")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K)
    args = parser.parse_args()

    chain = build_chain(args.model, args.top_k)
    if chain.retriever.db.count() == 0:
        print("The index is empty. Run: python scripts/build_index.py", file=sys.stderr)
        return 1

    history = ChatHistory()
    try:
        if args.question:
            answer(chain, args.question, history)
            return 0
        while True:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit", ""}:
                return 0
            answer(chain, question, history)
    except OllamaError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
