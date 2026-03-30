"""Tests for JWT authentication dependency."""

import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from src.auth import get_access_token, get_current_user_id
from src.config import get_settings

JWT_SECRET = "test-jwt-secret-for-auth-tests-minimum-32-bytes"

# Minimal app that uses the dependency
_test_app = FastAPI()


@pytest.fixture(autouse=True)
def _patch_jwt_secret():
    """Patch get_settings so the auth dependency uses our known secret."""
    mock_settings = MagicMock()
    mock_settings.supabase_jwt_secret = SecretStr(JWT_SECRET)
    with patch("src.auth.get_settings", return_value=mock_settings):
        yield


@_test_app.get("/protected")
async def protected(user_id: str = Depends(get_current_user_id)) -> dict[str, str]:
    """Return the authenticated user_id."""
    return {"user_id": user_id}


@_test_app.get("/token")
async def get_token(token: str = Depends(get_access_token)) -> dict[str, str]:
    """Return the raw access token."""
    return {"token": token}


# (Empty because the fixture above handles cleanup)


@pytest.mark.asyncio
async def test_missing_auth_header_returns_401() -> None:
    """Request without Authorization header returns 401."""
    transport = ASGITransport(app=_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/protected")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_invalid_token_returns_401() -> None:
    """Request with an invalid JWT returns 401."""
    transport = ASGITransport(app=_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/protected",
            headers={"Authorization": "Bearer not-a-valid-jwt"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_valid_token_returns_user_id() -> None:
    """Request with a valid JWT returns the user_id from 'sub' claim."""
    user_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    token = pyjwt.encode(
        {"sub": user_id, "aud": "authenticated"},
        JWT_SECRET,
        algorithm="HS256",
    )

    transport = ASGITransport(app=_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200
    assert response.json() == {"user_id": user_id}


@pytest.mark.asyncio
async def test_expired_token_returns_401() -> None:
    """Request with an expired JWT returns 401."""
    token = pyjwt.encode(
        {"sub": "some-user", "aud": "authenticated", "exp": int(time.time()) - 10},
        JWT_SECRET,
        algorithm="HS256",
    )

    transport = ASGITransport(app=_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_token_without_sub_returns_401() -> None:
    """JWT with no 'sub' claim returns 401."""
    token = pyjwt.encode(
        {"aud": "authenticated", "role": "authenticated"},
        JWT_SECRET,
        algorithm="HS256",
    )

    transport = ASGITransport(app=_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_access_token_returns_raw_jwt() -> None:
    """get_access_token returns the raw JWT string."""
    raw_token = pyjwt.encode(
        {"sub": "user-123", "aud": "authenticated"},
        JWT_SECRET,
        algorithm="HS256",
    )
    transport = ASGITransport(app=_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/token",
            headers={"Authorization": f"Bearer {raw_token}"},
        )
    assert response.status_code == 200
    assert response.json()["token"] == raw_token
