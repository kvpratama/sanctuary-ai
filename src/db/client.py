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
