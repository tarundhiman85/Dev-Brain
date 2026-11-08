# DevBrain Phase 2

DevBrain runs two local MCP servers over stdio. The GitHub server reads live
pull requests. The knowledge base server searches PostgreSQL directly and can
run while FastAPI is stopped.

## 1. Install

From the project root, use the Phase 1 Python environment.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

This installs the official [Python MCP SDK 1.x](https://py.sdk.modelcontextprotocol.io/v1/).

## 2. Configure

For terminal runs, put these settings in the project-root `.env` or set them
in the process environment. Environment variables take precedence.

- `GITHUB_TOKEN` is a GitHub personal access token. Select the relevant
  repositories and grant read access to Pull requests. Public repositories
  can also be queried without a token, subject to GitHub's unauthenticated rate limit.
- `DATABASE_URL` is your Phase 1 URI, such as
  `postgresql+psycopg://USER:PASSWORD@localhost:5432/devbrain`. URL-encode special
  characters in the credentials.
- `EMBEDDING_PROVIDER` must match the existing store. Use `local` for MiniLM
  or `openai` with `OPENAI_API_KEY` for OpenAI. The first KB server launch
  may download MiniLM. The OpenAI provider sends query text to OpenAI.

Keep credentials in local configuration. Start PostgreSQL and ingest documents
as described in [Phase 1](README_Phase1.md) before trying a search.

## 3. Test with MCP Inspector

Install Node.js for `npx`. Run each command from the project root in the
activated environment, then connect using stdio in the Inspector.

```powershell
npx -y @modelcontextprotocol/inspector python -m mcp_servers.github_mcp
npx -y @modelcontextprotocol/inspector python -m mcp_servers.kb_mcp
```

Under **Tools**, list the tools and try these calls.

| Tool | Example arguments |
| --- | --- |
| `list_recent_prs` | `{"repo_name":"octocat/Hello-World","limit":5}` |
| `fetch_pr_details` | `{"repo_name":"octocat/Hello-World","pr_number":1}` |
| `search_devbrain_kb` | `{"query":"How do we handle GitHub rate limits?","top_k":3}` |

Use a PR number from the list call if the example is unavailable. PR details
include up to 100 conversation comments, 100 inline review comments, and 100
changed files. The `has_more` fields flag additional pages. GitHub can omit
patches, for example for binary files. KB results include chunk content,
metadata, and cosine distance. Smaller distances indicate closer matches.

## 4. Connect Claude Desktop

1. Open **Settings > Developer > Edit Config** in Claude Desktop. On Windows
   the file is usually `%APPDATA%\Claude\claude_desktop_config.json`.
2. Merge both entries from [the template](claude_desktop_config_template.json)
   into your existing `mcpServers` object.
3. Set each `command` to the absolute path of the virtual environment's
   Python. On macOS, use `/absolute/project/path/.venv/bin/python`. Set each
   `PYTHONPATH` to the absolute project root so imports work from Claude's
   working directory. Replace the sample environment values and add
   `OPENAI_API_KEY` to the KB entry if using OpenAI.
4. Fully quit and restart Claude Desktop. Find the three tools and try the
   same calls as in the Inspector.

If tools are missing, check the Claude Desktop MCP logs and verify the Python
and project paths. If a call fails, check GitHub token access or the database
URI and embedding settings for the affected tool.
