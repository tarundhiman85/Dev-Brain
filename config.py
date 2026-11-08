"""Shared, validated configuration loaded from environment or .env."""
from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://devbrain:devbrain_local@localhost:5432/devbrain"
    embedding_provider: Literal["local", "openai", "ollama"] = "local"
    openai_api_key: SecretStr | None = None
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:11434")
    ollama_embedding_model: str = Field(default="nomic-embed-text", min_length=1)
    ollama_embedding_dimensions: int = Field(default=768, ge=1, le=2000)

    @property
    def dimensions(self) -> int:
        if self.embedding_provider == "ollama":
            return self.ollama_embedding_dimensions
        return 384 if self.embedding_provider == "local" else 1536

    @property
    def embedding_identity(self) -> str:
        if self.embedding_provider == "ollama":
            return f"ollama:{self.ollama_embedding_model}:{self.dimensions}:no-truncation-v1"
        # Include preprocessing so equal-sized but incompatible vectors cannot mix.
        return (
            "sentence-transformers/all-MiniLM-L6-v2:window-mean-v1"
            if self.embedding_provider == "local"
            else "openai:text-embedding-3-small:1536:v1"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
