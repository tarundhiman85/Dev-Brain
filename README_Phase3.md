# DevBrain Phase 3

The agent uses OpenAI function calling to choose among the Phase 2 knowledge
base and GitHub tools, plus Tavily web search. It runs a tool, gives its result
back to the model, and repeats until the model answers or the six-call budget
is exhausted. At the budget limit the model gets one final synthesis request.
[OpenAI function calling reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)

## Install and configure

From the project root, activate the Phase 1/2 virtual environment and install
the existing requirements. Phase 3 uses the OpenAI SDK and HTTPX already listed.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if needed, then set these values locally.

| Setting | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for chat, even with local embeddings. |
| `DEVBRAIN_AGENT_MODEL` | A Chat Completions model available to your account that supports function calling, for example `gpt-4.1-mini`. |
| `DATABASE_URL`, `EMBEDDING_PROVIDER` | Match the populated Phase 1 store. |
| `GITHUB_TOKEN` | Read access to the relevant repositories. Public repositories can use GitHub's unauthenticated access. |
| `TAVILY_API_KEY` | Enables web search. Without it, a web call returns a configuration error to the model. |

The agent sends the prompt, supplied conversation history, and tool observations
to OpenAI. These can include internal KB chunks, private PR descriptions, and
diffs. Tavily receives the search query. Use the endpoint with data you are
authorized to send to those services.

## Run

Start PostgreSQL and ingest notes using [Phase 1](README.md). Then launch the
agent API on its own port.

```powershell
docker compose up -d --wait
python -m uvicorn main_agent:app --host 127.0.0.1 --port 8001
```

The agent calls the Phase 2 tool functions in-process. Only the database needs
to be running locally. The agent endpoint has no authentication, so bind it to
localhost for personal use.

## Try a multi-step question

Replace `owner/repo` and `42` with a repository and PR you can access. Ingest
notes about the licensing service first so the question has local evidence.

```powershell
$url = 'http://127.0.0.1:8001/api/v1/agent/chat'
$prompt = 'What did I work on in the licensing service last week, and how does it compare to PR #42 in owner/repo?'
$first = Invoke-RestMethod -Uri $url -Method Post -ContentType 'application/json' -Body (
  @{ prompt = $prompt } | ConvertTo-Json
)
$first.answer
$first.trace | ConvertTo-Json -Depth 8

$second = Invoke-RestMethod -Uri $url -Method Post -ContentType 'application/json' -Body (
  @{
    prompt = 'Which changes in that PR were missing from my notes?'
    history = @(@{ prompt = $prompt; answer = $first.answer })
  } | ConvertTo-Json -Depth 8
)
$second.answer
```

For a web test, send a prompt such as “Find the current pgvector documentation
for HNSW index options and cite your sources.”

The API accepts up to eight previous prompt/answer pairs. The client supplies
them on each request, so different clients can keep separate conversations.
Each response contains an `answer` and a `trace` of tools, arguments, statuses,
and any errors or truncation. The trace records observable actions.

## Tool registration and tests

`agent/tools_bridge.py` uses the Phase 2 MCP input schemas for tool declarations
and validates arguments before invoking the corresponding async functions.
`agent/orchestrator.py` supplies routing instructions for the model. Historical
work goes to the KB, current PR details go to GitHub, and public technical
questions can go to Tavily. Missing repository context can trigger a follow-up
question. Tool results are bounded before they are passed to the model.

```powershell
python -m pytest -q
```
