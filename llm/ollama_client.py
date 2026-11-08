"""Ollama API calls for local chat and embeddings."""
from dataclasses import dataclass

import httpx


class OllamaUnavailable(RuntimeError):
    """The configured local model could not complete the request."""


@dataclass
class OllamaClient:
    http: httpx.AsyncClient
    model: str

    async def chat(self, messages: list[dict[str, str]]) -> str:
        try:
            response = await self.http.post(
                "api/chat",
                json={"model": self.model, "messages": messages, "stream": False,
                      "options": {"num_predict": 1024}},
            )
            response.raise_for_status()
            data = response.json()
            answer = data["message"]["content"].strip()
            if not answer or data.get("done") is not True or data.get("done_reason") == "length":
                raise ValueError("Incomplete Ollama response")
            return answer
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            raise OllamaUnavailable("Ollama chat failed. Check the service and installed chat model.") from None


def embed_texts(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    # Phase 1 ingestion and retrieval already run synchronously or in worker threads.
    try:
        with httpx.Client(base_url=base_url, timeout=httpx.Timeout(60, connect=3), trust_env=False) as http:
            response = http.post("api/embed", json={"model": model, "input": texts, "truncate": False})
            response.raise_for_status()
            return response.json()["embeddings"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise OllamaUnavailable("Ollama embeddings failed. Check the service and embedding model.") from None
