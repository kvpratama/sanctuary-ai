from unittest.mock import AsyncMock, MagicMock, patch

import src.db.client as client_module


async def test_get_supabase_client_returns_client():
    """get_supabase_client returns a Supabase AsyncClient instance."""
    client_module._supabase_client = None

    mock_settings = MagicMock()
    mock_settings.supabase_url.get_secret_value.return_value = (
        "https://test.supabase.co"
    )
    mock_settings.supabase_key.get_secret_value.return_value = "test-service-key"

    with (
        patch("src.db.client.get_settings", return_value=mock_settings),
        patch("src.db.client.acreate_client", new_callable=AsyncMock) as mock_create,
    ):
        mock_create.return_value = "fake-client"

        client = await client_module.get_supabase_client()
        assert client == "fake-client"

    client_module._supabase_client = None


async def test_get_supabase_client_uses_settings():
    """get_supabase_client reads URL and key from settings."""
    client_module._supabase_client = None

    mock_settings = MagicMock()
    mock_settings.supabase_url.get_secret_value.return_value = "https://my.supabase.co"
    mock_settings.supabase_key.get_secret_value.return_value = "my-key"

    with (
        patch("src.db.client.get_settings", return_value=mock_settings),
        patch("src.db.client.acreate_client", new_callable=AsyncMock) as mock_create,
    ):
        await client_module.get_supabase_client()

        mock_create.assert_called_once_with(
            "https://my.supabase.co",
            "my-key",
        )

    client_module._supabase_client = None


async def test_get_supabase_client_reuses_cached_client():
    """Subsequent calls return the same client without calling acreate_client again."""
    client_module._supabase_client = None

    mock_settings = MagicMock()
    mock_settings.supabase_url.get_secret_value.return_value = "https://x.supabase.co"
    mock_settings.supabase_key.get_secret_value.return_value = "key"

    with (
        patch("src.db.client.get_settings", return_value=mock_settings),
        patch("src.db.client.acreate_client", new_callable=AsyncMock) as mock_create,
    ):
        mock_create.return_value = "cached-client"

        first = await client_module.get_supabase_client()
        second = await client_module.get_supabase_client()

        assert first is second
        mock_create.assert_called_once()

    client_module._supabase_client = None


async def test_close_supabase_client_clears_cached_client():
    """close_supabase_client resets the module-level reference."""
    client_module._supabase_client = AsyncMock()

    await client_module.close_supabase_client()

    assert client_module._supabase_client is None


async def test_close_client_calls_individual_service_closers():
    """close_client calls close/aclose on postgrest, auth, and realtime."""
    mock_client = MagicMock()
    mock_client.postgrest = AsyncMock()
    mock_client.auth = AsyncMock()
    mock_client.realtime = AsyncMock()

    await client_module.close_client(mock_client)

    mock_client.postgrest.aclose.assert_called_once()
    mock_client.auth.close.assert_called_once()
    mock_client.realtime.close.assert_called_once()


async def test_get_authenticated_client_uses_anon_key() -> None:
    """get_authenticated_client creates a client with the anon key and sets the user session."""
    mock_settings = MagicMock()
    mock_settings.supabase_url.get_secret_value.return_value = (
        "https://test.supabase.co"
    )
    mock_settings.supabase_anon_key.get_secret_value.return_value = "test-anon-key"

    mock_client = AsyncMock()

    with (
        patch("src.db.client.get_settings", return_value=mock_settings),
        patch("src.db.client.acreate_client", new_callable=AsyncMock) as mock_create,
    ):
        mock_create.return_value = mock_client

        result = await client_module.get_authenticated_client("user-jwt-token")

        mock_create.assert_called_once_with(
            "https://test.supabase.co",
            "test-anon-key",
        )
        mock_client.auth.set_session.assert_called_once_with(
            access_token="user-jwt-token",
            refresh_token="",
        )
        assert result is mock_client
