"""Ingest trusted local Markdown and GitHub PR JSON exports."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, insert, text

from database import SessionLocal, engine, initialize_database
from embeddings import get_embedder
from models import DocumentChunk

logger = logging.getLogger(__name__)


class PullRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = Field(min_length=1)
    body: str | None = ""
    diff: str | None = ""
    comments: list[str] = Field(default_factory=list)
    url: str | None = None
    html_url: str | None = None
    number: int | None = None

    @field_validator("comments", mode="before")
    @classmethod
    def comment_bodies(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("comments must be a list")
        return [item.get("body") or "" if isinstance(item, dict) else item for item in value]


def documents(root: Path) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() == ".md":
            content = path.read_text(encoding="utf-8-sig")
            title = next((line.lstrip("# ").strip() for line in content.splitlines()
                          if line.startswith("# ")), path.stem)
            yield path.resolve().as_uri(), content, {
                "source_type": "markdown", "title": title, "path": str(path.resolve())}
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            entries = payload if isinstance(payload, list) else [payload]
            seen = set()
            for index, entry in enumerate(entries):
                pr = PullRequest.model_validate(entry)
                url = pr.html_url or pr.url
                source_id = url or f"{path.resolve().as_uri()}#pr-{pr.number if pr.number is not None else index}"
                if source_id in seen:
                    raise ValueError(f"Duplicate PR source in {path}: {source_id}")
                seen.add(source_id)
                content = f"# {pr.title}\n\n{pr.body or ''}\n\n## Diff\n{pr.diff or ''}\n\n## Comments\n" + "\n\n".join(pr.comments)
                yield source_id, content, {"source_type": "github_pr", "title": pr.title,
                                           "path": str(path.resolve()), "url": url, "number": pr.number}


def chunk_text(content: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""], disallowed_special=())
    return [chunk for chunk in splitter.split_text(content) if chunk.strip()]


def ingest_source(source_id: str, content: str, metadata: dict[str, Any]) -> int:
    chunks = chunk_text(content)
    # The source replacement is one transaction: embedding or insertion failures
    # leave its previously indexed version intact. Memory is bounded per source.
    with SessionLocal.begin() as session:
        lock_id = int.from_bytes(hashlib.sha256(source_id.encode()).digest()[:8], "big", signed=True)
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_id})
        session.execute(delete(DocumentChunk).where(DocumentChunk.source_id == source_id))
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            vectors = get_embedder().embed(batch)
            session.execute(insert(DocumentChunk), [
                {"source_id": source_id, "chunk_index": start + i, "content": chunk,
                 "source_metadata": {**metadata, "chunk_index": start + i}, "embedding": vector}
                for i, (chunk, vector) in enumerate(zip(batch, vectors, strict=True))])
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory must exist")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        initialize_database()
        count = 0
        for source_id, content, metadata in documents(args.directory):
            count += ingest_source(source_id, content, metadata)
            logger.info("Indexed %s", source_id)
        logger.info("Ingestion complete: %d chunks", count)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
