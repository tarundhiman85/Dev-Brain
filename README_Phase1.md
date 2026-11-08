# DevBrain — Phase 1

Local Markdown and GitHub PR JSON → recursive token chunks → embeddings →
PostgreSQL/pgvector → FastAPI semantic retrieval. This phase returns matching
chunks; it does not generate an LLM answer.

## 1. Prerequisites and install

Install Python 3.11+ and Docker Desktop (Linux containers). Run from this directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On Linux/macOS use `python3 -m venv .venv`, `source .venv/bin/activate`,
and `cp .env.example .env` instead. The remaining Python/Docker commands are the same.
Dependencies have compatible version ranges. `requirements.lock.txt` records the
validated Windows/Python 3.14 environment; use it on matching platforms with
`python -m pip install -r requirements.lock.txt`. Other Python/platform versions
should resolve `requirements.txt` and freeze their validated environment.

## 2. Start PostgreSQL

```sh
docker compose up -d --wait
```

PostgreSQL listens on localhost:5432 and stores data in a named volume. Set a
password in `.env`; keep `DATABASE_URL` consistent with the PostgreSQL variables
(URL-encode special characters in the URL). Changing the compose password does
not change credentials in an existing volume. The app/ingestion automatically
creates the vector extension, tables, and HNSW cosine index on first start. The
initial database user must have permission to create the extension. In a hosted
deployment, have an administrator enable it and use a dedicated application role.

## 3. Choose embeddings

The default `EMBEDDING_PROVIDER=local` downloads
`sentence-transformers/all-MiniLM-L6-v2` on first use (internet required; cached
afterward), runs on CPU, and produces 384-dimensional vectors. Its shorter model
context is handled by embedding token windows, taking a token-count-weighted
mean, and normalizing the final vector. This covers the entire 800-token chunk
without silent truncation, but long-chunk pooling can reduce retrieval precision.

For OpenAI set in `.env`:

```dotenv
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-key
```

This uses `text-embedding-3-small` with 1536 dimensions. Ingested content and
queries are sent to OpenAI and API charges apply. The OpenAI SDK supplies bounded
retries/timeouts. All returned vectors are checked for dimensions, finite values,
and nonzero norms before normalization.

**Choose the provider before initializing the database.** A persisted model and
preprocessing identity prevents mixing incompatible vectors. To change providers,
create a fresh database, update `DATABASE_URL`, and re-ingest. Schema creation is
bootstrap-only, not a migration system; use versioned migrations for later schema
changes. Do not change embedding settings while an existing process is running.

## 4. Ingest your documents

```sh
python ingest.py examples
python ingest.py /path/to/your/notes-and-exports
```

The scanner recursively loads UTF-8 `.md` files and `.json` PR exports. Keep the
input directory dedicated to these formats; unrelated JSON fails validation.
Markdown PR exports can also be ingested as ordinary Markdown. JSON may be one
PR object or a list:

```json
{
  "number": 42,
  "title": "Add API retries",
  "html_url": "https://github.com/owner/repository/pull/42",
  "body": "Retry transient failures with exponential backoff.",
  "diff": "- fetch()\n+ fetch_with_retry()",
  "comments": ["Bound retries", {"body": "Respect Retry-After"}]
}
```

`title` is required. `body`, `diff`, `comments`, `number`, and `url`/`html_url` are
optional. Sample data is synthetic; live GitHub fetching is outside this phase.
Source IDs use a PR URL when present, otherwise the absolute file URI plus PR
number/list position. Markdown uses its absolute file URI. Keep paths stable and
use PR URLs to make replacement reliable.

Chunks use `RecursiveCharacterTextSplitter`, `cl100k_base` token counting, size
800, and target overlap 100. Paragraphs, lines, and words are preferred split
boundaries; overlap can be shorter at those boundaries. Each chunk stores source
type, title, path/URL, and chunk index in JSONB (the Python ORM attribute is
`source_metadata` because SQLAlchemy reserves `metadata`).

Ingestion batches embeddings/inserts in groups of 32 and replaces each source in
one transaction under a source-specific advisory lock. Re-running does not
duplicate that source, and a failed replacement retains its old chunks. Empty
Markdown clears that source's chunks. Completed sources remain committed if a
later file fails; fix the error and rerun. Deleted/moved files and removed PRs are
not automatically pruned. Whole files and their chunk lists are held in memory;
split unusually large exports before ingestion.

## 5. Run and query

```sh
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open [Swagger UI](http://127.0.0.1:8000/docs) or run (Bash):

```sh
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"How do we handle GitHub rate limits?","top_k":3}'
```

PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/query -ContentType application/json -Body '{"query":"How do we handle GitHub rate limits?","top_k":3}'
```

Response shape:

```json
{
  "matches": [
    {
      "id": "7d53a91d-be34-4ed5-9faa-c8730f5f9501",
      "content": "# Add retry handling ...",
      "metadata": {"source_type":"github_pr","title":"Add retry handling","chunk_index":0},
      "distance": 0.18
    }
  ]
}
```

Smaller cosine distance means closer (typically 0–2); this is not a probability.
`top_k` defaults to 3 and must be an integer from 1 to 50. Queries must be
nonblank and at most 8000 characters. Empty databases return `{"matches":[]}`.
HNSW search is approximate; small datasets may use PostgreSQL's exact sequential
scan. Database queries have a 15-second statement timeout. Dependency failures
return 503 without exposing credentials/content in the HTTP response.

`GET /health` checks database connectivity. Startup validates schema configuration
and loads the embedding client/model. This personal local API has no
authentication: keep it bound to localhost; add authentication, TLS, quotas, and
an appropriately restricted database role before exposing it to other users.

## 6. Verify

```sh
python -m pytest -q
```

Offline tests cover parsing, token bounds, request validation, and empty results.
Set `RUN_MODEL_TESTS=1` to also download MiniLM and verify normalized embeddings
for both short inputs and inputs exceeding its context window (local provider only).
An optional database test checks cosine retrieval, replacement, and rollback using
synthetic embeddings. Run against a **disposable test database**, never a populated
knowledge base (synthetic vectors share its model configuration):

```powershell
$env:DATABASE_URL='postgresql+psycopg://devbrain:devbrain_local@localhost:5432/devbrain_test'
$env:RUN_DB_TESTS='1'
python -m pytest -q
Remove-Item Env:RUN_DB_TESTS
Remove-Item Env:DATABASE_URL
```

Create `devbrain_test` first with `docker compose exec postgres createdb -U devbrain devbrain_test`.
On Bash set these variables as command prefixes instead. To verify real model
loading and semantic quality, ingest `examples` and query the running API as above.

Stop with `docker compose down`; the database volume remains. Back up the volume
or use `pg_dump` before infrastructure/schema changes.

References: [pgvector SQLAlchemy integration](https://github.com/pgvector/pgvector-python#sqlalchemy),
[recursive token splitting](https://reference.langchain.com/python/langchain-text-splitters/base/TextSplitter/from_tiktoken_encoder),
[Sentence Transformer context limits](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html).
