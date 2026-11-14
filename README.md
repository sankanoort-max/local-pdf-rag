# Chat with your PDFs — local RAG

Ask questions about PDF files in a local folder. Everything runs on your
machine: Ollama for embeddings and the chat model, Chroma as the vector store,
Streamlit for the UI.

## 1. Install

1. Install Ollama from https://ollama.com and make sure it is running (`ollama serve`,
   or the desktop app).
2. Pull the two models (or any others — set their names in `.env`):
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.1:8b
   ```
3. Create a virtual environment and install packages (Python 3.10+):
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```

## 2. Add PDFs and index them

Copy PDFs into `data/pdfs/` (sub-folders are fine), or point `PDF_DIR` in `.env`
to another folder. Then either click **Index / Re-index PDFs** in the app, or run:

```
python scripts/build_index.py              # only new/changed files
python scripts/build_index.py --rebuild    # start from scratch
```

## 3. Ask questions

```
streamlit run app.py                         # web UI
python scripts/ask.py "What is the refund policy?"   # terminal
```

## 4. Run the tests (no Ollama needed)

```
pytest
```

## Project layout

| Folder / file | Job |
|---|---|
| `app.py` | Streamlit entry point; wires the UI to the rest |
| `config/settings.py` | All settings, read from `.env` |
| `ingestion/` | Load PDFs → clean text → chunk → run the indexing pipeline |
| `embeddings/` | Text → vectors with the Ollama embedding model |
| `vectorstore/` | Chroma wrapper: add, delete, search |
| `retrieval/` | Find the best chunks for a question (+ optional LLM reranker) |
| `llm/` | Ollama chat model wrapper and all prompts |
| `chains/` | The RAG flow: question → chunks → prompt → answer + sources |
| `memory/` | Chat history and follow-up question rewriting |
| `utils/` | Logging, file change tracking, small helpers |
| `ui/` | Sidebar, chat view, sources panel |
| `scripts/` | Command-line indexing and Q&A |
| `tests/` | Unit tests with fake models |

## Notes and common problems

- **"Cannot reach Ollama"** — start Ollama and check `OLLAMA_BASE_URL`.
- **"model not found"** — run `ollama pull <model>` for the name in `.env`.
- **Changing the embedding model** — the next indexing run rebuilds the whole
  index automatically, because vectors from different models can't be mixed.
  Questions always use the model that built the current index.
- **Scanned PDFs** — pages without a text layer are skipped and logged; add OCR
  (e.g. Tesseract) in `ingestion/pdf_loader.py` if you need them.
- **Tuning** — if answers miss information, try a larger `TOP_K`, a lower
  `SCORE_THRESHOLD`, smaller `CHUNK_SIZE`, or turn on the reranker.
- Logs are written to `logs/app.log`.
