from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIMENSIONS = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class Settings(BaseSettings):
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


settings = Settings()  # ty:ignore[missing-argument]
