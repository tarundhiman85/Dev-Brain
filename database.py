"""Connection pooling and serialized, idempotent schema initialization."""
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from config import get_settings
from models import Base, EmbeddingConfiguration

engine = create_engine(get_settings().database_url, pool_pre_ping=True,
                       connect_args={"connect_timeout": 10}, hide_parameters=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def initialize_database() -> None:
    settings = get_settings()
    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(918273645)"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(connection)
        row = connection.execute(select(EmbeddingConfiguration)).mappings().first()
        if row is None:
            connection.execute(EmbeddingConfiguration.__table__.insert().values(
                id=1, identity=settings.embedding_identity, dimensions=settings.dimensions))
        elif row["identity"] != settings.embedding_identity or row["dimensions"] != settings.dimensions:
            raise RuntimeError("Embedding configuration differs from this database. Use a fresh database and re-ingest.")
