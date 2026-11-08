"""Offline contract tests and an opt-in real PostgreSQL round trip."""
import os
from pathlib import Path

import pytest
import tiktoken
from fastapi.testclient import TestClient
from sqlalchemy import delete

import ingest
import main
import retrieval
from database import SessionLocal, initialize_database
from models import DocumentChunk


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="Downloads local embedding model")
def test_local_model_covers_long_inputs():
    import numpy as np
    from embeddings import get_embedder
    embedder = get_embedder()
    short = "Database connection pooling"
    long = short + (" retry requests with backoff" * 300)
    vectors = np.array(embedder.embed([short, long]))
    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert not np.allclose(vectors[0], vectors[1])


def test_recursive_chunks_have_token_bound_and_preserve_end():
    content = ("A paragraph about retrying database connections.\n\n" * 400) + "FINAL_SENTINEL"
    chunks = ingest.chunk_text(content)
    encoder = tiktoken.get_encoding("cl100k_base")
    assert len(chunks) > 1
    assert all(len(encoder.encode(chunk)) <= 800 for chunk in chunks)
    assert "FINAL_SENTINEL" in chunks[-1]


def test_export_parsing(tmp_path: Path):
    (tmp_path / "note.md").write_text("# Pooling\nUse a pool", encoding="utf-8")
    (tmp_path / "pr.json").write_text(
        '{"title":"Retries","body":null,"comments":[{"body":"Bound retries"}]}', encoding="utf-8")
    docs = list(ingest.documents(tmp_path))
    assert {doc[2]["source_type"] for doc in docs} == {"markdown", "github_pr"}
    assert "Bound retries" in docs[1][1]


def test_request_validation_and_empty_results(monkeypatch):
    class FakeEmbedder:
        def embed(self, texts):
            return [[1.0] + [0.0] * 383]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, statement):
            return self

        def all(self):
            return []

    monkeypatch.setattr(retrieval, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(retrieval, "SessionLocal", FakeSession)
    # Without a context manager TestClient intentionally skips startup/DB access.
    client = TestClient(main.app)
    for payload in ({"query": " "}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 51}):
        assert client.post("/api/v1/query", json=payload).status_code == 422
    response = client.post("/api/v1/query", json={"query": "pooling"})
    assert response.status_code == 200
    assert response.json() == {"matches": []}


@pytest.mark.skipif(os.getenv("RUN_DB_TESTS") != "1", reason="Requires a disposable pgvector database")
def test_real_vector_query_and_atomic_reingestion(monkeypatch):
    from config import get_settings

    class FakeEmbedder:
        def embed(self, texts):
            return [[1.0] + [0.0] * (get_settings().dimensions - 1) for _ in texts]

    source_id = "test://roundtrip"
    monkeypatch.setattr(ingest, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(retrieval, "get_embedder", lambda: FakeEmbedder())
    initialize_database()
    try:
        for content in ("Old note", "Updated note"):
            assert ingest.ingest_source(source_id, content, {"source_type": "markdown"}) == 1
        client = TestClient(main.app)
        response = client.post("/api/v1/query", json={"query": "note", "top_k": 50})
        assert response.status_code == 200
        matches = [item for item in response.json()["matches"] if item["content"] == "Updated note"]
        assert len(matches) == 1
        assert matches[0]["distance"] == pytest.approx(0, abs=1e-6)
        class BrokenEmbedder:
            def embed(self, texts):
                raise RuntimeError("provider unavailable")
        monkeypatch.setattr(ingest, "get_embedder", lambda: BrokenEmbedder())
        with pytest.raises(RuntimeError):
            ingest.ingest_source(source_id, "Failed replacement", {})
        response = client.post("/api/v1/query", json={"query": "note", "top_k": 50})
        assert any(item["content"] == "Updated note" for item in response.json()["matches"])
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(DocumentChunk).where(DocumentChunk.source_id == source_id))
