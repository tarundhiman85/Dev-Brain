"""Cover routing, Ollama fallback boundaries, and embedding compatibility."""
import asyncio
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import embeddings
from agent.orchestrator import ChatResponse
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
