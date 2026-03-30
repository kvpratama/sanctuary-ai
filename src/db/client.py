from src.config import get_settings
from supabase import AsyncClient, AsyncClientOptions, acreate_client

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


async def close_client(client: AsyncClient) -> None:
    """Close the underlying HTTP clients for a Supabase AsyncClient.

    Args:
        client: The Supabase client to close.
    """
    # AsyncClient in supabase-py v2.x doesn't have a top-level aclose().
    # We must close each service client that manages its own connection pool.
    if hasattr(client, "postgrest"):
        await client.postgrest.aclose()
    if hasattr(client, "auth"):
        await client.auth.close()
    if hasattr(client, "realtime"):
        await client.realtime.close()


async def close_supabase_client() -> None:
    """Close the shared Supabase client and reset the reference."""
    global _supabase_client
    if _supabase_client is not None:
        await close_client(_supabase_client)
        _supabase_client = None


async def get_authenticated_client(
    access_token: str, refresh_token: str | None = None
) -> AsyncClient:
    """Create a per-request Supabase client authenticated with the user's JWT.

    Uses the anon key (not the service-role key) so that Postgres RLS
    policies based on ``auth.uid()`` are enforced.

    Args:
        access_token: The user's Supabase JWT access token.
        refresh_token: Optional refresh token for session maintenance.

    Returns:
        An AsyncClient with the user's session set.
    """
    settings = get_settings()
    options = None
    if not refresh_token:
        options = AsyncClientOptions(
            headers={"Authorization": f"Bearer {access_token}"}
        )

    client = await acreate_client(
        settings.supabase_url.get_secret_value(),
        settings.supabase_anon_key.get_secret_value(),
        options=options,
    )

    if refresh_token:
        await client.auth.set_session(
            access_token=access_token, refresh_token=refresh_token
        )

    return client
