from typing import Any, AsyncGenerator

import jwt as pyjwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.config import get_settings
from src.db.client import close_client, get_authenticated_client
from supabase import AsyncClient

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_access_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """Return the raw access token from the Authorization header.

    Args:
        credentials: Bearer token from the Authorization header.

    Returns:
        The raw JWT string.

    Raises:
        HTTPException: 401 if the header is missing.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    return credentials.credentials


async def get_current_user_id(
    token: str = Depends(get_access_token),
) -> str:
    """Extract and verify JWT, returning the user ID.

    Args:
        token: Raw JWT access token string.

    Returns:
        The user ID (``sub`` claim) from the verified JWT.

    Raises:
        HTTPException: 401 if the token is missing, invalid, or expired.
    """
    settings = get_settings()
    try:
        payload = pyjwt.decode(
            token,
            settings.supabase_jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            audience="authenticated",
        )
    except pyjwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from e

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )
    return user_id


class AuthenticatedUser(BaseModel):
    """Authenticated user context containing ID and Supabase client.

    Attributes:
        id: The user's unique ID from the JWT.
        client: An authenticated Supabase client for this user.
    """

    id: str
    client: Any

    model_config = {"arbitrary_types_allowed": True}


async def get_authenticated_user(
    user_id: str = Depends(get_current_user_id),
    access_token: str = Depends(get_access_token),
) -> AsyncGenerator[AuthenticatedUser, None]:
    """FastAPI dependency that yields an authenticated user context.

    Combines user ID extraction, token retrieval, and Supabase client
    authentication into a single object. Uses a yield dependency to
    ensure the per-request Supabase client is closed after the request.

    Args:
        user_id: The verified user ID from the JWT.
        access_token: The raw JWT access token.

    Yields:
        An AuthenticatedUser object with both ID and auth_client.
    """
    client = await get_authenticated_client(access_token)
    try:
        yield AuthenticatedUser(id=user_id, client=client)
    finally:
        await close_client(client)
