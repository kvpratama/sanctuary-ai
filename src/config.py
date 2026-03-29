from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIMENSIONS = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class Settings(BaseSettings):
    """
    Application configuration and settings management.

    This module defines the configuration constants and Pydantic-based settings
    class for managing environment variables and application parameters.Application settings loaded from environment variables.

        Attributes:
            supabase_url: Supabase project URL.
            supabase_key: Supabase service key.
            supabase_jwt_secret: Supabase JWT secret for authentication.
            openai_api_key: OpenAI API key for LLM access.
            gemini_api_key: Google Gemini API key for embeddings.
            bookified_blob_read_write_token: Token for blob storage access.
            llm_model: Language model to use (default: gpt-4o-mini).
            llm_provider: LLM provider name (default: openai).
            llm_provider_base_url: Base URL for LLM provider API.
            embedding_model: Embedding model to use (default: gemini-embedding-001).
            cors_origins: List of allowed CORS origins.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required settings (loaded from environment variables)
    supabase_url: SecretStr
    supabase_key: SecretStr
    supabase_jwt_secret: SecretStr
    openai_api_key: SecretStr
    gemini_api_key: SecretStr
    bookified_blob_read_write_token: SecretStr

    # Optional settings with defaults
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    llm_provider_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "gemini-embedding-001"

    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings, constructed lazily on first call.

    Returns:
        The singleton Settings instance.
    """
    return Settings()  # ty:ignore[missing-argument]
