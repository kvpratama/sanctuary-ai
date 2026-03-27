from supabase import AsyncClient, acreate_client

from src.config import settings


async def get_supabase_client() -> AsyncClient:
    """Create and return an async Supabase client using service-role key."""
    return await acreate_client(
        settings.supabase_url.get_secret_value(),
        settings.supabase_key.get_secret_value(),
    )
