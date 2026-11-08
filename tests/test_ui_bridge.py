"""Verify the UI preflight and SSE event contract."""
import json

from fastapi.testclient import TestClient

from llm.router import RoutedResponse
from main_ui_bridge import app
from main_agent import get_router


def test_ui_stream_and_cors(monkeypatch):
    class FakeRouter:
        async def chat(self, request, progress):
            assert request.temperature == 0.7 and request.include_evidence
            progress("tool_start", {"tool": "search_devbrain_kb"})
            step = {"tool": "search_devbrain_kb", "status": "ok", "latency_ms": 10}
            progress("tool_end", step)
            return RoutedResponse(
                answer="Test answer", trace=[step], mode=request.mode, backend="cloud",
                model="test", route_reason="explicit_mode", latency_ms=20,
            )

    monkeypatch.setitem(app.dependency_overrides, get_router, FakeRouter)
    client = TestClient(app)
    origin = "http://127.0.0.1:3000"
    path = "/api/v1/agent/chat/stream"
    headers = {"Origin": origin, "Access-Control-Request-Method": "POST",
               "Access-Control-Request-Headers": "content-type"}
    response = client.options(path, headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert client.options(path, headers={**headers, "Origin": "https://other.example"}).status_code == 400

    response = client.post(path, headers={"Origin": origin}, json={
        "prompt": "Search my notes", "mode": "cloud", "temperature": 0.7, "include_evidence": True,
    })
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["access-control-allow-origin"] == origin
    frames = [frame.splitlines() for frame in response.text.split("\n\n") if frame.startswith("event:")]
    assert [frame[0] for frame in frames] == ["event: tool_start", "event: tool_end", "event: result"]
    assert json.loads(frames[-1][1][6:])["answer"] == "Test answer"
