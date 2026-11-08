# DevBrain Phase 4

The existing `POST /api/v1/agent/chat` endpoint now routes routine chat to
Ollama and tool-dependent or complex work to the OpenAI agent. The cloud
route reuses the Phase 3 knowledge base, GitHub, and Tavily tools.

## Set up Ollama

Install [Ollama](https://ollama.com/download) and start its desktop app or run
`ollama serve` if the service is stopped. Then pull a chat model. Pull the
embedding model if you plan to create an Ollama-backed knowledge base.

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
ollama list
```

Use the Phase 1–3 virtual environment and install `requirements.txt`.
Phase 4 uses the existing HTTPX and OpenAI SDK dependencies.

Copy `.env.example` to `.env` if needed, and configure the settings below.

| Setting | Default or purpose |
| --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434`. Point this at your trusted Ollama service. |
| `OLLAMA_CHAT_MODEL` | `llama3.2`. Use the exact installed tag, such as `qwen2.5-coder:7b`. |
| `OPENAI_API_KEY` | Required for cloud chat. |
| `DEVBRAIN_AGENT_MODEL` | Set to a tool-capable OpenAI model your account can access, such as `gpt-4o`. |
| `DEVBRAIN_ALLOW_CLOUD_FALLBACK` | Defaults to `false`. Set to `true` to allow Auto to send a failed local request and its history to OpenAI. |

Local chat can start without cloud credentials. Keep the API bound to localhost
for personal use. The cloud path can send conversation history and tool results
to OpenAI. Tavily receives search queries when configured and selected.
See [Phase 3](README_Phase3.md) for the tool credentials.

For an offline Ollama service, use installed local model tags, set
`OLLAMA_NO_CLOUD=1` in the Ollama service environment, and restart Ollama.
[Ollama FAQ](https://docs.ollama.com/faq)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main_agent:app --host 127.0.0.1 --port 8001
```

## Select a mode

| JSON `mode` | Behavior |
| --- | --- |
| `auto` | Default. Routes routine prompts to Ollama. Long context, code review or implementation cues, and cues for private or current data go to the cloud tool loop. Earlier user prompts also inform routing. |
| `local` | Uses Ollama with the supplied conversation history. A tool-capable model can list GitHub PRs and inspect PR details, including file changes. |

To try GitHub in Local Only, set `OLLAMA_CHAT_MODEL` to an installed tool-capable
tag such as `qwen2.5-coder:7b`, configure GitHub access, and restart the API.
Try `List recent pull requests in owner/repo` or `Show details for PR #42 in owner/repo`.
Explicit PR commands fetch and display GitHub data directly. Other PR
questions use Ollama tool calling. Both fetch GitHub data over the network.
| `cloud` | Uses the OpenAI agent and its tools. Requires cloud credentials. |

The Auto classifier uses rules and character count as a context-size estimate.
Use an explicit mode when routing matters, especially for sensitive input or a
tool-dependent request whose wording might evade the classifier. A selected
backend that cannot run returns HTTP 503. Auto's optional fallback applies to
local chat failures when cloud chat is configured.

```powershell
$uri = 'http://127.0.0.1:8001/api/v1/agent/chat'
$body = @{ mode = 'auto'; prompt = 'What is a Python tuple?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri $uri -ContentType 'application/json' -Body $body
```

For a cloud routing check, use `List the latest PRs in octocat/Hello-World`.
The response retains `answer` and `trace` and adds `mode`, `backend`, `model`,
`route_reason`, `fallback`, `latency_ms`, and cloud token `usage` when reported.
Continue a conversation with the Phase 3 `history` field.

## Use Ollama embeddings

Existing knowledge bases can keep their current embedding provider. To opt in,
create a **fresh database** on your pgvector-enabled PostgreSQL server. For the
default Docker Compose user:

```powershell
docker compose up -d --wait
docker compose exec postgres createdb -U devbrain devbrain_ollama
```

Set the following in `.env`. Use your actual database credentials.

```dotenv
DATABASE_URL=postgresql+psycopg://devbrain:devbrain_local@localhost:5432/devbrain_ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_DIMENSIONS=768
```

Run `python ingest.py examples` to initialize and populate the new database.
Restart the agent and any Phase 1 API or MCP KB processes with the same
settings. If Claude Desktop supplies environment overrides, update its KB
server settings too. The stored embedding identity includes the model tag and
dimension. Rebuild the store if the model behind that tag changes. Embedding
calls reject oversized inputs and retain the configured provider on failure
because vectors from different models are incompatible.
[Ollama embedding API](https://docs.ollama.com/api/embed)

## Benchmark latency and cost

Use the same prompts and history in both forced modes. Record cold startup
separately, warm both models, and run each case multiple times.

```powershell
$uri = 'http://127.0.0.1:8001/api/v1/agent/chat'
foreach ($mode in @('local', 'cloud')) {
  $body = @{ mode = $mode; prompt = 'Explain Python tuples in three sentences.' } | ConvertTo-Json
  1..5 | ForEach-Object {
    Invoke-RestMethod -Method Post -Uri $uri -ContentType 'application/json' -Body $body |
      Select-Object backend, model, latency_ms, usage
  }
}
```

Compare median and tail latency, and check answer quality for the same prompts.
For cloud calls, multiply the reported prompt and completion token totals by
your model's current per-token rates. The totals include every completion in
the tool loop. Account separately for caching rates, paid tools, and local
compute or electricity. For Auto, group measurements by the actual `backend`
and `fallback` fields.
