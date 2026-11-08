# DevBrain

DevBrain is a self-hosted developer assistant. Chat with a local Ollama model,
or use the cloud agent to search your indexed notes, inspect live GitHub pull
requests, and look up public technical information. The browser workspace shows
tool activity and expandable retrieved sources.

## Quick start on Windows

### 1. Install the prerequisites

- Python 3.11 or newer for the FastAPI backend.
- Node.js 20.9 or newer for the Next.js frontend.
- Ollama for local chat. Install Docker with Compose if you also want to run
  the PostgreSQL knowledge base.

Clone and configure the project from PowerShell:

```powershell
git clone https://github.com/tarundhiman85/Dev-Brain.git
cd Dev-Brain
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, use `python3 -m venv .venv`,
`source .venv/bin/activate`, and `cp .env.example .env`.
If you already have a `.env`, add missing settings from the template instead
of replacing it.

### 2. Start Ollama and the API

Start the Ollama desktop app, or run `ollama serve` in a separate terminal if
Ollama is stopped. Then pull the default chat model and start the API:

```powershell
ollama pull llama3.2
python -m uvicorn main_ui_bridge:app --host 127.0.0.1 --port 8001
```

If you already have another model installed, use `ollama list` to find its
exact tag and set `OLLAMA_CHAT_MODEL` in the root `.env` before starting the API.
Local chat works without PostgreSQL or an OpenAI key.

### 3. Start the frontend

In a second terminal at the repository root:

```powershell
cd frontend
npm ci
npm run dev
```

Open [DevBrain](http://127.0.0.1:3000), select **Local Only**, and send
`Explain Python tuples with a short example.` If PowerShell blocks the npm
script shim, use `npm.cmd` in place of `npm`.

The UI defaults to `http://127.0.0.1:8001` for API calls. For a different API URL,
copy `frontend/.env.example` to `frontend/.env.local` and edit
`NEXT_PUBLIC_DEVBRAIN_API_URL`. If the browser uses a different origin,
set `DEVBRAIN_UI_ORIGINS` in the root `.env` to a JSON array of exact origins
and restart the API. The default origins are `http://127.0.0.1:3000` and
`http://localhost:3000`.

## Enable the knowledge base and cloud tools

### 1. Start PostgreSQL and index sample documents

From the repository root, with the Python environment activated:

```powershell
docker compose up -d --wait
python ingest.py examples
```

The default embedding provider is local MiniLM and may download the model on
first use. Ingestion creates the pgvector schema and indexes the example
Markdown note and archived PR. To index your own data, run
`python ingest.py path/to/notes-and-pr-exports`. Keep the input directory
limited to Markdown and supported PR JSON exports.

Re-ingesting a source replaces its chunks. Deleted or moved files are not
automatically pruned. If you change the embedding provider or model, create a
fresh database and re-ingest to keep vectors compatible. The chat model and
embedding provider are configured independently.
[Ingestion format and database setup](README_Phase1.md)

### 2. Configure cloud chat and optional services

Set the required values in the root `.env`, then restart the chat API.

| Setting | Use |
| --- | --- |
| `OPENAI_API_KEY` | Required for cloud chat and the evaluation judge. Also required when using OpenAI embeddings. |
| `DEVBRAIN_AGENT_MODEL` | A tool-capable OpenAI model available to your account. Required for cloud chat. |
| `DATABASE_URL` | SQLAlchemy PostgreSQL URI. Match the Compose credentials, database name, and host port. |
| `EMBEDDING_PROVIDER` | `local` for MiniLM, `ollama` for Ollama embeddings, or `openai` for OpenAI embeddings. Must match the indexed database. |
| `GITHUB_TOKEN` | Enables authorized private-repository reads. Public repositories can use unauthenticated access within GitHub's limits. |
| `TAVILY_API_KEY` | Enables public web search. |
| `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL` | Local Ollama endpoint and installed chat model. |
| `DEVBRAIN_ALLOW_CLOUD_FALLBACK` | Defaults to `false`. When enabled, Auto can retry failed local chat through the configured cloud agent. |

### 3. Try a tool-backed conversation

In the browser, choose **Cloud Only** and send one of these prompts:

- **Knowledge base:** `Search my indexed notes for how we handle GitHub rate limits. Cite the archived PR.`
- **Live GitHub:** `List the three most recent PRs in octocat/Hello-World.`
- **Web:** `Search the web for the current pgvector HNSW index options and cite your sources.`

Open **Tool activity** to inspect arguments, status, and duration. Open
**Retrieved sources** to see source links or document excerpts and, for vector
matches, cosine similarity. A retrieved source may not be cited in the answer.

| Routing mode | Behavior |
| --- | --- |
| **Auto** | Uses prompt and history cues to route routine chat to Ollama and complex or tool-dependent work to the cloud agent. |
| **Local Only** | Uses Ollama for chat and supports read-only GitHub PR listing and details with a tool-capable model. The GitHub tool contacts GitHub directly. |
| **Cloud Only** | Uses the OpenAI agent and its available tools. |

The UI saves conversations in the browser and sends up to eight previous turns
with each prompt. Temperature follows the selected model's default unless you
enable a custom value. Cloud requests can send conversation history and retrieved
material to OpenAI. Tavily receives web search queries. Keep credentials in the
root `.env` and keep the unauthenticated API bound to loopback for personal use.

## Architecture

| Layer | Responsibility |
| --- | --- |
| [Next.js workspace](frontend/) | Stores chat sessions in the browser, sends prompts and routing preferences, and renders Markdown, traces, and sources. |
| [FastAPI chat API](main_agent.py) | Exposes JSON and SSE chat endpoints, validates requests, and permits configured browser origins. |
| [Hybrid router](llm/router.py) | Applies the requested mode and chooses Ollama or the cloud agent. |
| [Agent loop](agent/orchestrator.py) | Lets the cloud model select tools, runs up to six tool calls, and synthesizes an answer from their observations. |
| [Tool bridge](agent/tools_bridge.py) | Calls the same KB and GitHub functions used by the standalone MCP servers. Tavily supplies web search. |
| [Ingestion](ingest.py) and [retrieval](retrieval.py) | Chunks Markdown and archived PRs, stores embeddings in PostgreSQL, and searches pgvector by cosine distance. |
| [Evaluation harness](eval/eval_harness.py) | Runs golden questions against the chat API and scores retrieval, faithfulness, hallucination rate, and correctness. |

The browser sends one request per turn. The API routes it either to Ollama
for generation or to the cloud agent, which can call the knowledge base,
GitHub, and web tools. The streaming endpoint sends tool events as they happen
and delivers the synthesized answer in a final event. The MCP servers can
also run independently over stdio for clients such as Claude Desktop.

## Use the API directly

With the chat API running on port 8001:

```powershell
$body = @{ prompt = 'Explain Python tuples'; mode = 'local' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/api/v1/agent/chat -ContentType 'application/json' -Body $body
```

The JSON response includes the answer, selected backend and model, latency,
fallback status, and tool trace. Set `include_evidence` to `true` to include the
bounded observations the model saw. Browser clients can use
`POST /api/v1/agent/chat/stream` with the same request fields. Interactive
API documentation is at [port 8001 /docs](http://127.0.0.1:8001/docs).

To run semantic search independently, start the Phase 1 API on port 8000.
It exposes `POST /api/v1/query` and `GET /health`.

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

## Verify and troubleshoot

```powershell
python -m pytest -q
cd frontend
npm run typecheck
npm run build
```

| Symptom | Check |
| --- | --- |
| The browser cannot reach DevBrain | Check the API is running on port 8001 and the browser origin is in `DEVBRAIN_UI_ORIGINS`. |
| Ollama cannot answer | Run `ollama list` and check the service and `OLLAMA_CHAT_MODEL` tag. |
| Cloud chat returns 503 | Check that both `OPENAI_API_KEY` and `DEVBRAIN_AGENT_MODEL` are set, then restart the API. |
| Knowledge base search returns no useful matches | Check that ingestion targeted the same database and embedding configuration as the chat API. Try the bundled example query first. |
| GitHub or web tools fail | Verify repository access or the Tavily key, then inspect the tool error in the trace. |

To run the golden evaluation, follow the dedicated evaluation guide to create
and index a fixture database before starting the API against that database:

```powershell
python -m eval.eval_harness --mode cloud --label baseline --output .tmp/eval/baseline.json
```

Evaluation uses a cloud judge and requires an OpenAI key, even when comparing
against Local Only chat.

## Detailed guides

1. [Phase 1 · PostgreSQL, ingestion, and semantic search](README_Phase1.md)
2. [Phase 2 · GitHub and knowledge base MCP servers](README_Phase2.md)
3. [Phase 3 · Agent tools, API calls, and conversation history](README_Phase3.md)
4. [Phase 4 · Ollama, cloud routing, embeddings, and benchmarking](README_Phase4.md)
5. [Phase 5 · Golden dataset and evaluation reports](README_Phase5.md)
6. [Phase 6 · Next.js workspace, SSE, and CORS](README_Phase6_UI.md)
