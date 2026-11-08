# DevBrain Phase 6 · Chat UI

The Next.js workspace provides browser-local conversations, routing controls,
Markdown responses, code copy buttons, live tool activity, and expandable
retrieved sources. It uses the existing Phase 4 router and Phase 3 tool loop.

## 1. Start the API

From the repository root, activate the Python environment and start the UI
bridge. The existing REST endpoint is also available through this entry point.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main_ui_bridge:app --host 127.0.0.1 --port 8001
```

Use your existing root `.env` settings. For Local Only, start Ollama and set
`OLLAMA_CHAT_MODEL` to an installed tag. For cloud chat, set `OPENAI_API_KEY`
and `DEVBRAIN_AGENT_MODEL`. Knowledge base tools also need the initialized
PostgreSQL database and its matching embedding settings. GitHub and web tools
use the settings in [Phase 3](README_Phase3.md). See
[Phase 4](README_Phase4.md) for model setup and fallback behavior.

## 2. Start the frontend

Install Node.js 20.9 or newer. From another terminal at the repository root:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open [DevBrain](http://127.0.0.1:3000). If PowerShell blocks the npm script
shim, use `npm.cmd` in place of `npm`.

`NEXT_PUBLIC_DEVBRAIN_API_URL` in `frontend/.env.local` defaults to
`http://127.0.0.1:8001`. If the frontend uses a different hostname or port,
set `DEVBRAIN_UI_ORIGINS` in the backend's root `.env` to a JSON array of exact
browser origins, for example `["http://127.0.0.1:3001"]`. Restart the
corresponding process after editing its environment. The default origins are
`http://localhost:3000` and `http://127.0.0.1:3000`.

For a production build, run `npm run build` and then `npm start` inside
`frontend`. Next.js embeds the public API URL at build time. Keep the API
on loopback for personal use because it has no authentication.

## 3. Use the workspace

- **Auto** chooses a backend. **Local Only** uses Ollama and can list and inspect
  GitHub PRs with a tool-capable model.
  **Cloud Only** uses the cloud agent and can call GitHub, KB, and web tools.
- Temperature uses the model default unless you turn off **Model default**.
  Some cloud models require their default temperature.
- Tool activity updates as each call starts and completes. Open a tool row to
  inspect its arguments, status, and duration. The answer appears when the
  agent finishes synthesis.
- Retrieved sources are expandable. Knowledge base cards display cosine
  similarity as `1 - distance`; GitHub and web results have no vector score.
  These cards show retrieved material, which may include sources the answer
  did not cite.
- Conversations and source previews are saved in this browser. The workspace
  retains up to 30 conversations and 50 completed turns per conversation.
  Up to eight previous turns are sent with a prompt.

## API transport

`POST /api/v1/agent/chat` still returns a JSON response for integrations and the
evaluation harness. The browser uses `POST /api/v1/agent/chat/stream` with the
same request schema and receives SSE events named `tool_start`, `tool_end`,
`result`, and `error`. For proxies, disable buffering on the streaming route.

Run backend checks with `python -m pytest -q` and frontend checks with
`npm run build`.
