"""Shared, validated configuration loaded from environment or .env."""
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://devbrain:devbrain_local@localhost:5432/devbrain"
    embedding_provider: Literal["local", "openai"] = "local"
    openai_api_key: SecretStr | None = None

    @property
    def dimensions(self) -> int:
        return 384 if self.embedding_provider == "local" else 1536

    @property
    def embedding_identity(self) -> str:
        # Include preprocessing so equal-sized but incompatible vectors cannot mix.
        return (
            "sentence-transformers/all-MiniLM-L6-v2:window-mean-v1"
            if self.embedding_provider == "local"
            else "openai:text-embedding-3-small:1536:v1"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
