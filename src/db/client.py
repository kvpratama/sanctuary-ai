from src.config import get_settings
from supabase import AsyncClient, acreate_client

_supabase_client: AsyncClient | None = None


async def get_supabase_client() -> AsyncClient:
    """Return the shared async Supabase client, creating it on first call.

    Returns:
        The cached AsyncClient instance.
    """
    global _supabase_client
    if _supabase_client is None:
        settings = get_settings()
        _supabase_client = await acreate_client(
            settings.supabase_url.get_secret_value(),
            settings.supabase_key.get_secret_value(),
        )
    return _supabase_client


async def close_supabase_client() -> None:
    """Reset the cached Supabase client reference."""
    global _supabase_client
    _supabase_client = None


async def get_authenticated_client(access_token: str) -> AsyncClient:
    """Create a per-request Supabase client authenticated with the user's JWT.

    Uses the anon key (not the service-role key) so that Postgres RLS
    policies based on ``auth.uid()`` are enforced.

    Args:
        access_token: The user's Supabase JWT access token.

    Returns:
        An AsyncClient with the user's session set.
    """
    settings = get_settings()
    client = await acreate_client(
        settings.supabase_url.get_secret_value(),
        settings.supabase_anon_key.get_secret_value(),
    )
    await client.auth.set_session(access_token=access_token, refresh_token="")
    return client
