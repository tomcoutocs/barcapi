# Veterinary RAG API

Production-oriented FastAPI service for ingesting trusted veterinary knowledge (PDF, DOCX, TXT, HTML), chunking with token-aware sentence boundaries, embedding with OpenAI, storing vectors in **Pinecone** or **Weaviate**, and retrieving ranked, deduplicated context for LLM grounding.

## Architecture

`Ingestion → Chunking → Embedding → Vector storage → Retrieval API`

- **Ingestion** (`app/ingestion/`): `load_pdf`, `load_docx`, `load_txt`, optional HTML stripper; normalized document schema with metadata (`type`, optional `species`, `topic`).
- **Chunking** (`app/chunking/`): ~500–800 tokens per chunk (tiktoken `cl100k_base`), ~100–150 token overlap, sentence-safe splits; oversized sentences split on token windows.
- **Embeddings** (`app/embeddings/`): `EmbeddingProvider` abstraction; `OpenAIEmbeddingProvider` (e.g. `text-embedding-3-large`).
- **Vector store** (`app/vectorstore/`): pluggable `VectorStore` with `upsert_chunks` and `query_similar` + metadata filters (`species`, `type`, `topic`).
- **Retrieval** (`app/retrieval/`): query embedding, over-fetch, **lightweight hybrid** re-rank (vector score + keyword overlap), **source-quality weights** (manual/guideline favored), **length-based confidence**, optional **cross-encoder re-rank** (`sentence-transformers`), **deduplication**, LRU **query cache**.

## Requirements

- **Python 3.11 or newer** (3.10+ may work; this repo is developed against 3.11. Older runtimes such as 3.7 cannot install current FastAPI/OpenAI stacks.)
- OpenAI API key
- Pinecone index **or** Weaviate cluster (local or cloud)

### Pinecone index setup

Create an index whose **dimension** matches your embedding model (3072 for full `text-embedding-3-large`, or match `OPENAI_EMBEDDING_DIMENSIONS` if you use Matryoshka-style reduced dimensions). Use **cosine** metric. Metadata fields used in filters: `species`, `type`, `topic` (only include keys that exist on your vectors).

### Weaviate

With `VECTOR_BACKEND=weaviate`, the app creates the collection schema on first run (custom vectors, no built-in vectorizer).

## Environment

Configuration is read from dotenv files in this order (later overrides earlier): `vet-rag-api/.env`, **repo root** `.env`, `vet-rag-api/.env.local`, **repo root** `.env.local`. That matches a Next.js app using **`.env.local` at the monorepo root** (e.g. `barc/.env.local`): put the RAG variables there alongside your existing keys. Process environment variables still override dotenv.

Optional: set **`VET_RAG_ENV_FILE`** to an absolute path to append one more dotenv file (loaded last).

See `vet-rag-api/.env.example` for variable names. Set at least:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required |
| `OPENAI_EMBEDDING_MODEL` | Default `text-embedding-3-large` |
| `OPENAI_EMBEDDING_DIMENSIONS` | Optional; must match Pinecone index |
| `VECTOR_BACKEND` | `pinecone` (default) or `weaviate` |
| `PINECONE_API_KEY` | For Pinecone |
| `PINECONE_INDEX_NAME` | Index name |
| `PINECONE_HOST` | Optional; explicit host URL (recommended in production) |
| `WEAVIATE_URL` | e.g. `http://localhost:8080` or cloud URL |
| `WEAVIATE_API_KEY` | For Weaviate Cloud |
| `WEAVIATE_CLASS_NAME` | Default `VetChunk` |
| `RETRIEVAL_TOP_K` | Default retrieval count (5–8 typical) |
| `QUERY_CACHE_ENABLED` | `true` / `false` |
| `RERANK_ENABLED` | `true` requires `pip install sentence-transformers` |
| `OPENAI_CHAT_MODEL` | Chat model for `POST /chat` (default `gpt-4o-mini`) |
| `AGENT_EVAL_LOG_PATH` | JSONL path for agent interaction logs |

## Install & run

On Windows, **`python` often points to an old install (e.g. 3.7)**. This project needs **3.11+**. Use the **`py` launcher** so the venv is created with the right runtime:

```powershell
cd vet-rag-api
py -0p
```

Pick any **3.11+** entry from the list (e.g. `-V:3.13`), then create the venv with that tag:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If `py -3.12` says *No suitable Python runtime found*, you do not have 3.12 installed—use **`py -0p`** and run `py -3.13 -m venv .venv` (or `py -3.14`, etc.) instead.

Confirm `python --version` shows **3.11.x or newer** after activating the venv.

### If `pip` fails with `No matching distribution found for uvicorn>=0.24.0`

That almost always means the venv was created with **Python 3.7 or older**. Remove `.venv`, run `py -0p`, then `py -3.13 -m venv .venv` (or another **3.11+** version from the list). If none appear, install Python from [python.org](https://www.python.org/downloads/windows/).

Open `http://localhost:8000/docs` for interactive OpenAPI.

## Dog assistant agent (`POST /chat`)

Safety-first orchestration for **dog-focused** owner questions on top of the **existing** retriever (no RAG rewrite).

| Module | Role |
|--------|------|
| `app/agent/interpreter.py` | `interpret_query()` — symptoms, duration, severity flags, toxins |
| `app/agent/triage.py` | `classify_triage()`, `generate_followup_questions()` |
| `app/agent/safety.py` | System prompt + escalation copy |
| `app/agent/rag_client.py` | `retrieve_context(query)` → `list[dict]` (wraps `RetrievalService`) |
| `app/agent/responder.py` | OpenAI JSON response with retrieved context |
| `app/agent/formatter.py` | Stable JSON shape for clients |
| `app/agent/evaluator.py` | JSONL logging + triage test cases |

**Request:** `POST /chat` with `{"message":"..."}`.

**Response:** `{ triage_level, summary, possible_causes, what_to_monitor, recommended_action, urgency_message }`.

Set **`OPENAI_CHAT_MODEL`** (default `gpt-4o-mini`). Logs append to **`AGENT_EVAL_LOG_PATH`** (default `logs/agent_interactions.jsonl`).

**Triage-only tests (no LLM):** `python -m scripts.run_agent_eval`

## Scrape → ingest pipeline (optional)

Modular scrapers (Merck, AVMA, AAHA/WSAVA PDFs) feed **`ingest_documents()`**, which reuses the same chunk/embed/vector flow as the HTTP API. See **[SCRAPING.md](./SCRAPING.md)** for layout, `pipeline_main.py`, and compliance notes.

## API

### `POST /chat`

JSON body: `{ "message": "my dog is vomiting and won’t eat" }`. Returns the structured agent response (see **Dog assistant agent** above). Uses the same `OPENAI_API_KEY` as embeddings plus **`OPENAI_CHAT_MODEL`**.

### `POST /ingest`

Multipart form:

- `file` (required): PDF, DOCX, TXT, or HTML
- `document_id` (optional): stable id; UUID generated if omitted
- `doc_type`: `manual` | `journal` | `guideline` | `drug` | `sop`
- `species`, `topic`, `source_label` (optional)

### `POST /query`

JSON body (see `examples/query_example.json`):

```json
{
  "query": "my dog is vomiting and lethargic",
  "species": "dog",
  "document_type": null,
  "topic": null,
  "top_k": 8
}
```

Response:

```json
{
  "query": "my dog is vomiting and lethargic",
  "context": [
    {
      "text": "...",
      "source": "<document_id>",
      "score": 0.0,
      "metadata": { "type": "guideline", "species": "dog", "topic": "...", "vector_score": 0.0, "keyword_overlap": 0.0 }
    }
  ]
}
```

### `GET /health`

Liveness check.

## Example CLI ingest

```powershell
cd vet-rag-api
.\.venv\Scripts\Activate.ps1
python -m scripts.ingest_example .\samples\chapter.txt --doc-type manual --species dog --topic nutrition
```

## Example query (curl)

```powershell
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d "@examples/query_example.json"
```

## Safety & quality notes

This service **retrieves** evidence; it does **not** diagnose or prescribe. Downstream LLMs should cite `source` / `metadata.source_label` and obey clinical governance. Retrieval applies source-type weights and deduplication to reduce redundancy; filters help scope species and document class.

## Optional: cross-encoder re-ranking

```powershell
pip install sentence-transformers
```

Set `RERANK_ENABLED=true` and optionally `RERANK_MODEL` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`).

## License

Use and deploy per your organization’s policies for veterinary software and third-party content (Merck, AVMA, AAHA, etc.).
