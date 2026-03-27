from unittest.mock import AsyncMock, patch


async def test_get_supabase_client_returns_client():
    """get_supabase_client returns a Supabase AsyncClient instance."""
    with patch("db.client.settings") as mock_settings:
        mock_settings.supabase_url.get_secret_value.return_value = (
            "https://test.supabase.co"
        )
        mock_settings.supabase_key.get_secret_value.return_value = "test-service-key"

        with patch("db.client.acreate_client", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "fake-client"

            from db.client import get_supabase_client

            client = await get_supabase_client()
            assert client == "fake-client"


async def test_get_supabase_client_uses_settings():
    """get_supabase_client reads URL and key from settings."""
    with (
        patch("db.client.settings") as mock_settings,
        patch("db.client.acreate_client", new_callable=AsyncMock) as mock_create,
    ):
        mock_settings.supabase_url.get_secret_value.return_value = (
            "https://my.supabase.co"
        )
        mock_settings.supabase_key.get_secret_value.return_value = "my-key"

        from db.client import get_supabase_client

        await get_supabase_client()

        mock_create.assert_called_once_with(
            "https://my.supabase.co",
            "my-key",
        )
