"""Cover routing, Ollama fallback boundaries, and embedding compatibility."""
import asyncio
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import embeddings
from agent.orchestrator import ChatResponse, ToolStep
from config import Settings
from llm.ollama_client import OllamaClient, OllamaUnavailable
from llm.router import CloudUnavailable, LLMRouter, RoutedRequest, classify


@pytest.mark.parametrize(("prompt", "backend"), [
    ("What is a Python tuple?", "local"),
    ("What did I work on in licensing?", "cloud"),
    ("Compare PR #42 with my notes", "cloud"),
    ("Refactor this parser", "cloud"),
    ("Here is the code:\n```python\nprint(1)\n```", "cloud"),
    ("x" * 2001, "cloud"),
])
def test_auto_classifies(prompt, backend):
    assert classify(RoutedRequest(prompt=prompt))[0] == backend


def test_local_chat_sends_history_without_contacting_cloud():
    def ollama(request):
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert [message["role"] for message in payload["messages"]] == [
            "system", "user", "assistant", "user",
        ]
        return httpx.Response(200, json={"done": True, "message": {"content": "A tuple is immutable."}})

    async def run():
        async with httpx.AsyncClient(base_url="http://localhost:11434", transport=httpx.MockTransport(ollama)) as http:
            router = LLMRouter(OllamaClient(http, "llama3.2"), None)
            return await router.chat(RoutedRequest(
                mode="local", prompt="Tell me more about PR #42",
                history=[{"prompt": "What is a tuple?", "answer": "It is a sequence."}],
            ))

    result = asyncio.run(run())
    assert result.backend == "local"
    assert result.answer == "A tuple is immutable."
    assert result.trace == []


def test_fallback_requires_auto_and_opt_in():
    def offline(request):
        raise httpx.ConnectError("Connection refused", request=request)

    async def run():
        async with httpx.AsyncClient(base_url="http://localhost:11434", transport=httpx.MockTransport(offline)) as http:
            cloud = AsyncMock()
            cloud.model = "test-cloud"
            cloud.chat.return_value = ChatResponse(answer="Cloud result", trace=[])
            router = LLMRouter(OllamaClient(http, "llama3.2"), cloud)
            prompt = "Define a tuple"
            with pytest.raises(OllamaUnavailable):
                await router.chat(RoutedRequest(prompt=prompt))
            router.allow_cloud_fallback = True
            with pytest.raises(OllamaUnavailable):
                await router.chat(RoutedRequest(prompt=prompt, mode="local"))
            cloud.chat.assert_not_awaited()
            fallback = await router.chat(RoutedRequest(prompt=prompt))
            assert fallback.backend == "cloud" and fallback.fallback
            forced = await router.chat(RoutedRequest(prompt=prompt, mode="cloud"))
            assert forced.backend == "cloud" and not forced.fallback
            router.cloud = None
            with pytest.raises(CloudUnavailable):
                await router.chat(RoutedRequest(prompt=prompt, mode="cloud"))

    asyncio.run(run())


@pytest.mark.parametrize(("prompt", "tool", "arguments"), [
    ("List the 3 most recent pull requests in team/repo.", "list_recent_prs",
     {"repo_name": "team/repo", "limit": 3}),
    ("Show details for PR #42 in https://github.com/team/repo.", "fetch_pr_details",
     {"repo_name": "team/repo", "pr_number": 42}),
])
def test_local_github_commands_use_read_only_tool_directly(prompt, tool, arguments):
    cloud = AsyncMock()
    local_github = AsyncMock()
    pr = {"number": 42, "html_url": "https://github.com/team/repo/pull/42",
          "title": "Test PR", "state": "open"}
    observation = {"pull_requests": [pr]} if tool == "list_recent_prs" else pr
    local_github.run_tool.return_value = (json.dumps(observation), ToolStep(tool=tool, status="ok"))
    local = Mock(model="local-test-model")
    router = LLMRouter(local, cloud, allow_cloud_fallback=True, local_github=local_github)
    request = RoutedRequest(mode="local", prompt=prompt)
    response = asyncio.run(router.chat(request))
    assert response.backend == "local" and response.route_reason == "local_github_tools"
    assert "PR #42" in response.answer and response.trace[0].tool == tool
    local_github.run_tool.assert_awaited_once_with(tool, json.dumps(arguments), False, None, explicit_command=True)
    local_github.chat.assert_not_awaited()
    cloud.chat.assert_not_awaited()


def test_local_web_search_uses_only_explicit_query():
    cloud = AsyncMock()
    local = Mock(model="local-test-model")
    local_tools = AsyncMock()
    observation = {"results": [{"url": "https://example.com/mcp", "title": "MCP", "content": "A result"}]}
    local_tools.run_tool.return_value = (json.dumps(observation), ToolStep(tool="web_search", status="ok"))
    router = LLMRouter(local, cloud, local_github=local_tools)
    request = RoutedRequest(
        mode="local", prompt="search the web for latest recommend practices for mcp",
        history=[{"prompt": "List recent PRs in private/repo", "answer": "A private PR"}],
    )
    result = asyncio.run(router.chat(request))
    assert result.backend == "local" and "[MCP](https://example.com/mcp)" in result.answer
    assert result.trace[0].tool == "web_search"
    local_tools.run_tool.assert_awaited_once_with(
        "web_search", json.dumps({"query": "latest recommend practices for Model Context Protocol (MCP)"}),
        False, None, explicit_command=True,
    )
    local_tools.chat.assert_not_awaited()
    cloud.chat.assert_not_awaited()


def test_ollama_embeddings_share_validation_and_model_identity(monkeypatch):
    settings = Settings(embedding_provider="ollama", ollama_embedding_model="test-embed", ollama_embedding_dimensions=2)
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    mock_embed = Mock(return_value=[[3, 4]])
    monkeypatch.setattr(embeddings, "embed_texts", mock_embed)
    embedder = embeddings.Embedder()
    assert embedder.embed(["hello"]) == [[0.6, 0.8]]
    mock_embed.assert_called_once_with(str(settings.ollama_base_url), "test-embed", ["hello"])
    assert settings.embedding_identity == "ollama:test-embed:2:no-truncation-v1"
    mock_embed.return_value = [[1, 2, 3]]
    with pytest.raises(ValueError, match="invalid vectors"):
        embedder.embed(["hello"])
