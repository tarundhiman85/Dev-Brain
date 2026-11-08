"""Exercise the agent protocol with offline provider responses."""
import asyncio
import json

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from agent import orchestrator
from agent.tools_bridge import ToolBridge
from llm.ollama_client import OllamaClient
from llm.router import LLMRouter
from main_agent import app, get_router


def completion(message, finish_reason="stop"):
    return httpx.Response(200, json={
        "id": "test", "object": "chat.completion", "created": 0, "model": "test",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    })


def tool_call(name, arguments, call_id="call_1"):
    return completion({"role": "assistant", "tool_calls": [{
        "id": call_id, "type": "function", "function": {"name": name, "arguments": arguments},
    }]}, "tool_calls")


def test_chat_routes_multiple_tools_and_keeps_context(monkeypatch):
    from mcp_servers import github_mcp

    async def fake_pr(repo_name: str, pr_number: int):
        assert (repo_name, pr_number) == ("team/licensing", 42)
        return {"body": "Adds a cache"}

    monkeypatch.setattr(github_mcp, "fetch_pr_details", fake_pr)
    requests = []

    def model(request):
        payload = json.loads(request.content)
        requests.append(payload)
        messages = payload["messages"]
        assert messages[1]["content"] == "Which repo?"
        assert messages[2]["content"] == "team/licensing"
        if len(requests) == 1:
            assert {item["function"]["name"] for item in payload["tools"]} == {
                "fetch_pr_details", "list_recent_prs", "search_devbrain_kb", "web_search",
            }
            return tool_call("fetch_pr_details", '{"repo_name":"team/licensing","pr_number":42}')
        if len(requests) == 2:
            assert messages[-1]["tool_call_id"] == "call_1"
            assert json.loads(messages[-1]["content"]) == {"body": "Adds a cache"}
            return tool_call("web_search", '{"query":"cache invalidation"}', "call_2")
        assert messages[-1]["tool_call_id"] == "call_2"
        assert json.loads(messages[-1]["content"])["results"][0]["url"] == "https://example.com"
        return completion({"role": "assistant", "content": "PR #42 adds a cache."})

    def tavily(request):
        assert request.url == "https://api.tavily.com/search"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content)["query"] == "cache invalidation"
        return httpx.Response(200, json={"results": [
            {"title": "Caching", "url": "https://example.com", "content": "Example"},
        ]})

    provider = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(model)))
    web = httpx.AsyncClient(transport=httpx.MockTransport(tavily))
    agent = orchestrator.Orchestrator(provider, "test", ToolBridge(web, "test-key"))
    app.dependency_overrides[get_router] = lambda: LLMRouter(OllamaClient(web, "test-local"), agent)
    try:
        client = TestClient(app)
        assert client.post("/api/v1/agent/chat", json={"prompt": " "}).status_code == 422
        response = client.post("/api/v1/agent/chat", json={
            "prompt": "Compare PR #42 to public caching guidance",
            "history": [{"prompt": "Which repo?", "answer": "team/licensing"}],
        })
        assert response.status_code == 200
        assert response.json()["answer"] == "PR #42 adds a cache."
        assert response.json()["backend"] == "cloud"
        assert [step["status"] for step in response.json()["trace"]] == ["ok", "ok"]
    finally:
        app.dependency_overrides.pop(get_router)

        async def close():
            await provider.close()
            await web.aclose()

        asyncio.run(close())


def test_chat_synthesizes_after_failed_tool_calls(monkeypatch):
    monkeypatch.setattr(orchestrator, "MAX_TOOL_CALLS", 2)

    def model(request):
        payload = json.loads(request.content)
        last = payload["messages"][-1]
        if last["role"] == "user":
            return tool_call("search_devbrain_kb", "[]")
        if last["tool_call_id"] == "call_1":
            assert "valid JSON object" in last["content"]
            return tool_call("list_recent_prs", '{"repo_name":"bad-repo"}', "call_2")
        assert payload["tool_choice"] == "none"
        assert "Invalid tool arguments" in last["content"]
        return completion({"role": "assistant", "content": "I need more context."})

    async def run():
        async with (
            AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(model))) as client,
            httpx.AsyncClient() as web,
        ):
            agent = orchestrator.Orchestrator(client, "test", ToolBridge(web, None))
            return await agent.chat(orchestrator.ChatRequest(prompt="What changed?"))

    response = asyncio.run(run())
    assert response.answer == "I need more context."
    assert [step.status for step in response.trace] == ["error", "error"]
