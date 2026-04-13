from functools import lru_cache

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIMENSIONS = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class JudgeConfig(BaseModel):
    """Configuration for a single LLM judge in the evaluation jury.

    Attributes:
        model: Model name (e.g. "gpt-4o", "claude-sonnet").
        provider: LangChain provider id (e.g. "openai", "anthropic").
        api_key_field: Name of a SecretStr field on Settings holding the API key.
        base_url: Optional custom API endpoint. Empty string means use provider default.
    """

    model: str
    provider: str
    api_key_field: str
    base_url: str = ""


class Settings(BaseSettings):
    """
    Application configuration and settings management.

    This module defines the configuration constants and Pydantic-based settings
    class for managing environment variables and application parameters.

    Application settings loaded from environment variables.

        Attributes:
            supabase_url: Supabase project URL.
            supabase_key: Supabase service key.
            supabase_anon_key: Supabase anon key.
            openai_api_key: OpenAI API key for LLM access.
            gemini_api_key: Google Gemini API key for embeddings.
            bookified_blob_read_write_token: Token for blob storage access.
            blob_storage_origin: Origin URL for blob storage.
            llm_model: Language model to use (default: gpt-4o-mini).
            llm_provider: LLM provider name (default: openai).
            llm_provider_base_url: Base URL for LLM provider API.
            grading_llm_model: Language model to use for internal relevance grading (default: gpt-4o-mini).
            embedding_model: Embedding model to use (default: gemini-embedding-001).
            min_similarity: Minimum similarity threshold for retrieval.
            cors_origins: List of allowed CORS origins.
            eval_llm_model: Language model to use for evaluations (default: gpt-4o-mini).
            eval_llm_provider: LLM provider for evaluations (default: openai).
            eval_llm_provider_base_url: Base URL for evaluation LLM provider API.
            eval_llm_api_key: API key for evaluation LLM (optional, falls back to openai_api_key).
            rag_strategy: RAG pipeline strategy to use (default: naive_rag).
            eval_jury_judges: List of judge configurations for jury-of-judges evaluation (optional).
            langsmith_api_key: LangSmith API key for tracing (optional).
            langsmith_project: LangSmith project name (default: default).
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
    supabase_anon_key: SecretStr
    openai_api_key: SecretStr
    gemini_api_key: SecretStr
    bookified_blob_read_write_token: SecretStr

    # Blob storage origin allowlist (scheme + host) for safe token forwarding
    blob_storage_origin: str = "https://public.blob.storage.com"

    # Optional settings with defaults
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    llm_provider_base_url: str = "https://api.openai.com/v1"
    grading_llm_model: str = "gpt-4o-mini"
    embedding_model: str = "gemini-embedding-001"

    min_similarity: float = 0.6

    cors_origins: list[str] = ["http://localhost:3000"]

    # Eval-specific LLM configuration (used by src/eval/ only)
    eval_llm_model: str = "gpt-4o-mini"
    eval_llm_provider: str = "openai"
    eval_llm_provider_base_url: str = "https://api.openai.com/v1"
    eval_llm_api_key: SecretStr | None = None
    cerebras_api_key: SecretStr | None = None

    # RAG strategy selection for experimentation
    rag_strategy: str = "naive_rag"

    # Jury-of-judges configuration (optional JSON array of judge configs)
    eval_jury_judges: list[JudgeConfig] | None = None

    # LangSmith configuration
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "default"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings, constructed lazily on first call.

    Returns:
        The singleton Settings instance.
    """
    return Settings()  # ty:ignore[missing-argument]
