from unittest.mock import AsyncMock, MagicMock, patch


async def test_get_supabase_client_returns_client():
    """get_supabase_client returns a Supabase AsyncClient instance."""
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

        from src.db.client import get_supabase_client

        client = await get_supabase_client()
        assert client == "fake-client"


async def test_get_supabase_client_uses_settings():
    """get_supabase_client reads URL and key from settings."""
    mock_settings = MagicMock()
    mock_settings.supabase_url.get_secret_value.return_value = "https://my.supabase.co"
    mock_settings.supabase_key.get_secret_value.return_value = "my-key"

    with (
        patch("src.db.client.get_settings", return_value=mock_settings),
        patch("src.db.client.acreate_client", new_callable=AsyncMock) as mock_create,
    ):
        from src.db.client import get_supabase_client

        await get_supabase_client()

        mock_create.assert_called_once_with(
            "https://my.supabase.co",
            "my-key",
        )
