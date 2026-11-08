"""DevBrain semantic retrieval API; synchronous routes run in FastAPI's pool."""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from database import SessionLocal, engine, initialize_database
from embeddings import get_embedder
from retrieval import QueryRequest, QueryResponse, search_chunks

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(initialize_database)
        await run_in_threadpool(get_embedder)
        yield
    finally:
        engine.dispose()


app = FastAPI(title="DevBrain", version="0.1.0", lifespan=lifespan)


@app.post("/api/v1/query", response_model=QueryResponse)
def query_store(request: QueryRequest) -> QueryResponse:
    try:
        return search_chunks(request)
    except Exception:
        logger.exception("Semantic query failed")
        raise HTTPException(status_code=503, detail="Retrieval unavailable; check server logs") from None


@app.get("/health")
def health() -> dict[str, str]:
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable") from None
