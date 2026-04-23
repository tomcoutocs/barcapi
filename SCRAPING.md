# Veterinary scrape → ingest pipeline

This adds **scraping, cleaning, metadata enrichment, and deduplication** on top of the **existing** RAG path (`normalize_document` → `chunk_document` → OpenAI embeddings → Pinecone/Weaviate). It does **not** replace chunking, embeddings, or the vector store.

## Layout

| Path | Role |
|------|------|
| `app/scrapers/merck.py` | Merck Veterinary Manual HTML |
| `app/scrapers/avma.py` | Generic HTML (`scrape_html_resource`) + AVMA helper |
| `app/scrapers/guidelines.py` | AAHA / WSAVA / AVMA **PDFs** (bytes → `load_pdf_bytes`) |
| `app/scrapers/scrape_dispatch.py` | `scrape_one` / `run_all_scrapers` URL routing |
| `app/utils/cleaner.py` | `clean_text` |
| `app/utils/metadata.py` | `enrich_metadata` (topic, urgency, authority, `species=dog`) |
| `app/utils/deduper.py` | SHA-256 fingerprint + **≥300 words** gate |
| `app/pipeline/ingest_documents.py` | **`ingest_documents([{text, metadata}])`** → existing upsert |
| `app/pipeline/process_and_ingest.py` | `process_and_ingest(scraped_docs)` |
| `pipeline_main.py` | Example CLI: seed URLs → scrape → ingest |

## Setup

Same Python venv and `.env.local` as the main API (OpenAI + Pinecone/Weaviate). Install scrape deps:

```powershell
cd vet-rag-api
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

1. **`pipeline_main.py`** crawls **Merck Veterinary Manual → Dog owners** via BFS from `MERCK_CRAWL_SEEDS` (default hub: `/dog-owners`), up to **`MERCK_CRAWL_MAX_ARTICLES`** article URLs (paths with at least `/dog-owners/<section>/<page>`). Each page is fetched **once** for both link discovery and text extraction.
2. Add optional **`ADDITIONAL_URLS`** in `pipeline_main.py` for AVMA / AAHA / WSAVA (same as before).
3. Tune in **`.env.local`**:

| Variable | Meaning |
|----------|---------|
| `MERCK_CRAWL_MAX_ARTICLES` | Stop after this many articles (default `2500`) |
| `MERCK_CRAWL_MAX_VISITS` | Safety cap on BFS page fetches (default `35000`) |
| `MERCK_CRAWL_DELAY_S` | Pause between HTTP requests (default `1.0`) |

```powershell
python pipeline_main.py
# e.g. larger batch:
# MERCK_CRAWL_MAX_ARTICLES=600 python pipeline_main.py
```

**Cost:** each ingested document creates multiple embedding API calls and vector upserts. Hundreds of pages → significant OpenAI + Pinecone usage.

Logs show crawl progress, skipped documents (short/duplicate), fetch failures (after 3 retries), and per-document upserts.

## Ingest document shape

`ingest_documents` expects:

```json
{
  "text": "...",
  "metadata": {
    "source": "canonical URL",
    "title": "...",
    "type": "manual | guideline | org",
    "species": "dog",
    "topic": "...",
    "urgency": "low | medium | high | emergency",
    "authority_weight": 0.95
  }
}
```

`type: "org"` is stored in the vector DB as **`journal`** (existing `DocumentType` union). Extra fields (`title`, `urgency`, `authority_weight`) are stored on chunks for retrieval metadata.

## Legal / ethical use

- Merck, AAHA, WSAVA, and AVMA content is **copyrighted**. Use only what your organization is licensed to index, or public summaries you have rights to use.
- Implement **rate limiting** (`delay_s` in `run_all_scrapers`) and honor **robots.txt**.
- This code is for **internal/clinical tooling** with proper governance, not for bulk redistribution of publisher content.

## Optional: Playwright

Not installed by default. Add Playwright only if a target page requires a full browser; then extend a scraper to drive Chromium and still return `{url, title, raw_text, source_type}`.

## Weaviate note

If you already had a **`VetChunk`** class **before** `title` / `urgency` / `authority_weight` existed, either create a **new** class name via `WEAVIATE_CLASS_NAME` or recreate the collection so the schema includes those properties.
