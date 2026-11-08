"""Shared semantic retrieval for FastAPI and MCP."""
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, text

from config import get_settings
from database import SessionLocal
from embeddings import get_embedder
from models import DocumentChunk, EmbeddingConfiguration


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=3, ge=1, le=50, strict=True)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace characters")
        return value.strip()


class Match(BaseModel):
    id: UUID
    content: str
    metadata: dict[str, Any]
    distance: float


class QueryResponse(BaseModel):
    matches: list[Match]


class EmbeddingConfigurationError(RuntimeError):
    pass


def search_chunks(request: QueryRequest, *, validate_configuration: bool = False) -> QueryResponse:
    with SessionLocal() as session:
        session.execute(text("SET LOCAL statement_timeout = '15s'"))
        if validate_configuration:
            configuration = session.get(EmbeddingConfiguration, 1)
            settings = get_settings()
            if (configuration is None or configuration.identity != settings.embedding_identity
                    or configuration.dimensions != settings.dimensions):
                raise EmbeddingConfigurationError(
                    "DATABASE_URL and EMBEDDING_PROVIDER must match the initialized Phase 1 store."
                )
        session.execute(text("SET LOCAL hnsw.ef_search = 100"))
        vector = get_embedder().embed([request.query])[0]
        distance = DocumentChunk.embedding.cosine_distance(vector)
        rows = session.execute(select(DocumentChunk, distance.label("distance"))
                               .order_by(distance).limit(request.top_k)).all()
        return QueryResponse(matches=[Match(id=chunk.id, content=chunk.content,
                                            metadata=chunk.source_metadata, distance=float(score))
                                      for chunk, score in rows])
