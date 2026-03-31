"""Tests for the application configuration and settings."""

import pytest
from pydantic import SecretStr

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIMENSIONS, Settings


def test_configuration_constants():
    """Test that configuration constants have expected values."""
    assert EMBEDDING_DIMENSIONS == 768
    assert CHUNK_SIZE == 1000
    assert CHUNK_OVERLAP == 200


def test_settings_loads_from_env(monkeypatch):
    """Test that Settings loads values from environment variables.

    Args:
        monkeypatch: Pytest fixture for modifying environment variables.
    """
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("BOOKIFIED_BLOB_READ_WRITE_TOKEN", "test-blob-token")

    settings = Settings(_env_file=None)  # ty:ignore[missing-argument,unknown-argument]

    assert settings.supabase_url.get_secret_value() == "https://test.supabase.co"
    assert settings.supabase_key.get_secret_value() == "test-key"
    assert settings.openai_api_key.get_secret_value() == "test-openai-key"
    assert settings.gemini_api_key.get_secret_value() == "test-gemini-key"
    assert (
        settings.bookified_blob_read_write_token.get_secret_value() == "test-blob-token"
    )


def test_settings_default_values(monkeypatch):
    """Test that optional settings have correct default values.

    Args:
        monkeypatch: Pytest fixture for modifying environment variables.
    """
    # Set required env vars
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("BOOKIFIED_BLOB_READ_WRITE_TOKEN", "test-blob-token")

    # Override any .env file values for optional settings
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    settings = Settings(_env_file=None)  # ty:ignore[missing-argument,unknown-argument]

    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_provider == "openai"
    assert settings.llm_provider_base_url == "https://api.openai.com/v1"
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.cors_origins == ["http://localhost:3000"]


def test_settings_custom_cors_origins(monkeypatch):
    """Test that CORS origins can be customized.

    Args:
        monkeypatch: Pytest fixture for modifying environment variables.
    """
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("BOOKIFIED_BLOB_READ_WRITE_TOKEN", "test-blob-token")
    monkeypatch.setenv("CORS_ORIGINS", '["http://example.com", "http://test.com"]')

    settings = Settings(_env_file=None)  # ty:ignore[missing-argument,unknown-argument]
    assert settings.cors_origins == ["http://example.com", "http://test.com"]


def test_settings_secret_str_type(monkeypatch):
    """Test that sensitive fields are stored as SecretStr.

    Args:
        monkeypatch: Pytest fixture for modifying environment variables.
    """
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("BOOKIFIED_BLOB_READ_WRITE_TOKEN", "test-blob-token")

    settings = Settings(_env_file=None)  # ty:ignore[missing-argument,unknown-argument]

    assert isinstance(settings.supabase_url, SecretStr)
    assert isinstance(settings.supabase_key, SecretStr)
    assert isinstance(settings.openai_api_key, SecretStr)
    assert isinstance(settings.gemini_api_key, SecretStr)
    assert isinstance(settings.bookified_blob_read_write_token, SecretStr)


def test_settings_loads_supabase_anon_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings loads supabase_anon_key from environment."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "test-anon-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("BOOKIFIED_BLOB_READ_WRITE_TOKEN", "test-blob")

    settings = Settings(_env_file=None)  # ty:ignore[missing-argument,unknown-argument]
    assert settings.supabase_anon_key.get_secret_value() == "test-anon-key"
    assert isinstance(settings.supabase_anon_key, SecretStr)
