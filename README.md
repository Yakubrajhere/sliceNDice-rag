# RAG Project (Local, Ollama + ChromaDB)

Ask questions over your own PDF / TXT / DOCX files. Fully local: embeddings and LLM run on Ollama, vectors persist in ChromaDB.

## Pipeline

```
data/ -> document_loader -> text_splitter -> embeddings -> vector_db (vectorstore/)
                                                               |
question -> embeddings -> retriever -> prompt_template -> llm -> answer + sources
```

## Models

| Role       | Model                 | Why |
|------------|-----------------------|-----|
| LLM        | `qwen2.5:7b-instruct` | Best instruction-following / grounded answers among installed models at 7B; fast |
| Embeddings | `nomic-embed-text`    | Dedicated embedding model (the chat models above are not suited for retrieval) |

Optional upgrade: set `LLM_MODEL=gemma3:12b` in `.env` (better quality, ~8 GB, slower).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

ollama pull nomic-embed-text    # one-time
cp .env.example .env            # Windows: copy .env.example .env
```

## Usage

```bash
python ingest.py --reset        # build the vector store from data/
python app.py                   # chat with your documents
```

Re-run `python ingest.py` after adding files (safe to repeat; unchanged chunks are not duplicated). Use `--reset` if you removed or edited files.

## Configuration (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_MODEL` | `qwen2.5:7b-instruct` | Ollama chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Characters per chunk / overlap |
| `TOP_K` | `4` | Chunks retrieved per question |
| `TEMPERATURE` | `0.1` | Lower = more factual |
| `NUM_CTX` | `4096` | LLM context window |

> Changing `EMBEDDING_MODEL` requires `python ingest.py --reset`.

## Notes
- `data/` ships with small sample files; replace with your own.
- `.env` is git-ignored; `.env.example` is committed.

## Repository Structure
```text
├── data/                  # Drop reference PDFs/TXTs/DOCXs here
├── src/
│   ├── document_loader.py # Processes raw files
│   ├── text_splitter.py   # Custom text chunking engine
│   ├── embeddings.py      # Ollama vector generation wrapper
│   ├── vector_db.py       # Custom persistent Chroma integration 
│   ├── retriever.py       # Core similarity gating mechanics
│   ├── prompt_template.py # Core system grounding prompts
│   └── llm.py             # Local model orchestration
├── .env.example           # Environment blueprint
├── app.py                 # Chat application entrypoint
├── ingest.py              # Ingestion automation entrypoint
└── requirements.txt       # App requirements
```