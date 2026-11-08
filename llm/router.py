"""Select a backend before sending conversation data to a model."""
import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal
from urllib.parse import quote
from openai import OpenAIError

from agent.orchestrator import ChatRequest, ChatResponse, Orchestrator, ToolProgress
from llm.ollama_client import OllamaClient, OllamaUnavailable

RoutingMode = Literal["auto", "local", "cloud"]
Backend = Literal["local", "cloud"]

_TOOL_CUES = re.compile(
    r"\b(?:github|pull requests?|pr(?:s)?|diffs?|knowledge base|kb|internal|past work|"
    r"my notes|our notes|our repo(?:sitory)?|my repo(?:sitory)?|what did (?:i|we)|"
    r"last week|latest|today|current(?:ly)?|recent(?:ly)?|web search|search the web|"
    r"look up|browse|cite sources|verify online)\b",
    re.IGNORECASE,
)
_GITHUB_CUES = re.compile(r"\b(?:github|pull requests?|prs?)\b", re.IGNORECASE)
_WEB_COMMAND = re.compile(r"^(?:please\s+)?(?:search the web for|web search for|look up on the web)\s+(.+)$", re.IGNORECASE)
_REPOSITORY = re.compile(r"(?<![\w./-])(?:https://github\.com/)?([A-Za-z0-9-]+/[A-Za-z0-9_.-]+)")
_PR_NUMBER = re.compile(r"\b(?:pull request|pr)\s*#?(\d{1,10})\b|/pull/(\d{1,10})\b", re.IGNORECASE)
_PR_COMMAND = re.compile(r"^(?:please\s+)?(?:list|show|get|fetch|summarize)\b", re.IGNORECASE)
_RECENT_PRS = re.compile(r"\b(?:pull requests|prs)\b", re.IGNORECASE)
_PR_LIMIT = re.compile(r"\b(\d{1,3})\s+(?:(?:most\s+)?recent\s+|latest\s+)?(?:pull requests|prs)\b", re.IGNORECASE)
_COMPLEX_CUES = re.compile(
    r"```|\b(?:compare|refactor\w*|debug\w*|root cause|code review|review (?:my|this|the) code|"
    r"security audit|multi[- ]step|step[- ]by[- ]step|trade[- ]offs?|architecture|"
    r"design (?:a|an|the)|implement|write (?:a|an|the) (?:function|script|class|program))\b",
    re.IGNORECASE,
)


class RoutedRequest(ChatRequest):
    mode: RoutingMode = "auto"


class RoutedResponse(ChatResponse):
    mode: RoutingMode
    backend: Backend
    model: str
    route_reason: str
    fallback: bool = False
    latency_ms: float


class CloudUnavailable(RuntimeError):
    pass


def explicit_github_tool(prompt: str) -> tuple[str, dict[str, str | int]] | None:
    if not _PR_COMMAND.match(prompt):
        return None
    repos = {match.group(1).rstrip(".") for match in _REPOSITORY.finditer(prompt)}
    if len(repos) != 1:
        return None
    repo = repos.pop()
    if number := _PR_NUMBER.search(prompt):
        return "fetch_pr_details", {"repo_name": repo, "pr_number": int(number.group(1) or number.group(2))}
    if _RECENT_PRS.search(prompt):
        count = _PR_LIMIT.search(prompt)
        return "list_recent_prs", {"repo_name": repo, "limit": int(count.group(1)) if count else 5}
    return None


def github_answer(tool: str, result: dict[str, Any], repo: str) -> str:
    def link(pr: dict[str, Any]) -> str:
        return f"[PR #{pr['number']}]({pr['html_url']})"

    if tool == "list_recent_prs":
        prs = result.get("pull_requests", [])
        if not prs:
            return f"No pull requests found in `{repo}`."
        rows = [f"- {link(pr)} · **{pr['state']}** · {pr['title']}" for pr in prs]
        return "\n".join([f"Recent pull requests in `{repo}`", "", *rows])

    sections = [
        f"### {link(result)} · {result['title']}",
        f"**Status:** {'merged' if result.get('merged') else result['state']}",
        (result.get("body") or "No description provided.")[:5000],
    ]
    files = result.get("files", [])
    if files:
        rows = [f"- `{file['filename']}` (+{file['additions']} / −{file['deletions']})" for file in files[:15]]
        sections.append("\n".join(["**Changed files**", *rows]))
    comments = [*result.get("comments", []), *result.get("review_comments", [])]
    if comments:
        rows = [f"- {(comment.get('body') or '')[:500]}" for comment in comments[:5]]
        sections.append("\n".join(["**Comments**", *rows]))
    return "\n\n".join(sections)


def web_answer(result: dict[str, Any]) -> str:
    rows = []
    for item in result.get("results", []):
        url = item["url"]
        if not url.lower().startswith(("https://", "http://")):
            continue
        title = re.sub(r"([\\\[\]])", r"\\\1", item["title"])
        link = quote(url, safe=":/?&=%#@+;,~!$*'-._")
        rows.append(f"- [{title}]({link}) — {' '.join(item['content'].split())}")
    return "\n\n".join(["Web search results", "\n".join(rows)]) if rows else "No web results found."


def classify(request: RoutedRequest) -> tuple[Backend, str]:
    if request.mode != "auto":
        return request.mode, "explicit_mode"
    prompts = "\n".join([*(turn.prompt for turn in request.history), request.prompt])
    if _TOOL_CUES.search(prompts):
        return "cloud", "tool_or_fresh_data_cue"
    if _COMPLEX_CUES.search(prompts):
        return "cloud", "complexity_cue"
    # Character count is a cheap context-size proxy with no tokenizer dependency.
    if len(prompts) + sum(len(turn.answer) for turn in request.history) > 2000:
        return "cloud", "long_context"
    return "local", "routine_prompt"


@dataclass
class LLMRouter:
    local: OllamaClient
    cloud: Orchestrator | None
    allow_cloud_fallback: bool = False
    local_github: Orchestrator | None = None

    async def chat(self, request: RoutedRequest, progress: ToolProgress | None = None) -> RoutedResponse:
        started = perf_counter()
        backend, reason = classify(request)
        model = self.local.model
        fallback = False
        github_request = request.mode == "local" and any(
            _GITHUB_CUES.search(prompt) for prompt in [request.prompt, *(turn.prompt for turn in request.history)]
        )
        web_request = _WEB_COMMAND.fullmatch(request.prompt) if request.mode == "local" else None
        if web_request and self.local_github is not None:
            query = web_request.group(1).strip()
            if not re.search(r"\bmodel context protocol\b", query, re.IGNORECASE):
                query = re.sub(r"\bmcp\b", "Model Context Protocol (MCP)", query, flags=re.IGNORECASE)
            result = await self.run_local_command(("web_search", {"query": query}), request.include_evidence, progress)
            reason, model = "local_web_search", "Web search tool"
        elif github_request and self.local_github is not None:
            reason = "local_github_tools"
            if command := explicit_github_tool(request.prompt):
                result = await self.run_local_command(command, request.include_evidence, progress)
                model = "GitHub tool"
            else:
                try:
                    result = await self.local_github.chat(request, progress=progress)
                except OpenAIError:
                    raise OllamaUnavailable(
                        "Local GitHub chat needs a running Ollama server and an installed model that supports tools."
                    ) from None
        elif backend == "local":
            messages = [{
                "role": "system",
                "content": (
                    "You are DevBrain, a local developer assistant. Answer using your knowledge "
                    "and the provided conversation. You have no access to external tools. "
                    "State when you lack the information to answer."
                ),
            }]
            for turn in request.history:
                messages.extend([
                    {"role": "user", "content": turn.prompt},
                    {"role": "assistant", "content": turn.answer},
                ])
            messages.append({"role": "user", "content": request.prompt})
            try:
                result = ChatResponse(answer=await self.local.chat(messages, request.temperature), trace=[])
            except OllamaUnavailable:
                if request.mode != "auto" or not self.allow_cloud_fallback or self.cloud is None:
                    raise
                backend, fallback = "cloud", True

        if backend == "cloud":
            if self.cloud is None:
                raise CloudUnavailable("Cloud chat requires OPENAI_API_KEY and DEVBRAIN_AGENT_MODEL.")
            result = await self.cloud.chat(request, progress=progress)
            model = self.cloud.model

        return RoutedResponse(
            **result.model_dump(), mode=request.mode, backend=backend, model=model,
            route_reason=reason, fallback=fallback, latency_ms=round((perf_counter() - started) * 1000, 2),
        )

    async def run_local_command(
        self, command: tuple[str, dict[str, str | int]], include_evidence: bool,
        progress: ToolProgress | None,
    ) -> ChatResponse:
        assert self.local_github is not None
        name, arguments = command
        observation, step = await self.local_github.run_tool(
            name, json.dumps(arguments), include_evidence, progress, explicit_command=True,
        )
        if step.status == "error":
            answer = step.error or "Tool request failed."
        else:
            result = json.loads(observation)
            answer = web_answer(result) if name == "web_search" else github_answer(name, result, str(arguments["repo_name"]))
            if step.truncated:
                answer += "\n\nSome tool output was truncated. Open the linked sources for complete details."
        return ChatResponse(answer=answer, trace=[step])
