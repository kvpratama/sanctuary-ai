"""Tests for JWT authentication dependency."""

import time
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from src.auth import get_access_token, get_current_user_id

_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_public_key = _private_key.public_key()

# Minimal app that uses the dependency
_test_app = FastAPI()


@pytest.fixture(autouse=True)
def _patch_auth_dependencies() -> Generator:
    """Patch the JWT signing key discovery so tests don't hit the network."""
    mock_jwks_client = MagicMock()
    mock_signing_key = MagicMock()
    mock_signing_key.key = _public_key
    mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

    with (
        patch("src.auth._get_jwks_client", return_value=mock_jwks_client),
    ):
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
async def test_jwks_client_error_returns_401() -> None:
    """Network failure fetching JWKS keys returns 401, not 500."""
    mock_jwks_client = MagicMock()
    mock_jwks_client.get_signing_key_from_jwt.side_effect = pyjwt.PyJWKClientError(
        "Network error"
    )

    with patch("src.auth._get_jwks_client", return_value=mock_jwks_client):
        transport = ASGITransport(app=_test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/protected",
                headers={"Authorization": "Bearer some-token"},
            )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_valid_token_returns_user_id() -> None:
    """Request with a valid JWT returns the user_id from 'sub' claim."""
    user_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    token = pyjwt.encode(
        {"sub": user_id, "aud": "authenticated"},
        _private_key,
        algorithm="RS256",
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
        _private_key,
        algorithm="RS256",
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
        _private_key,
        algorithm="RS256",
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
        _private_key,
        algorithm="RS256",
    )
    transport = ASGITransport(app=_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/token",
            headers={"Authorization": f"Bearer {raw_token}"},
        )
    assert response.status_code == 200
    assert response.json()["token"] == raw_token


@pytest.mark.asyncio
async def test_get_authenticated_user_yields_and_closes() -> None:
    """get_authenticated_user yields a client and closes it in finally."""
    from src.auth import AuthenticatedUser, get_authenticated_user

    user_id = "test-user"
    token = "test-token"
    mock_client = MagicMock()

    with (
        patch("src.auth.get_current_user_id", return_value=user_id),
        patch("src.auth.get_access_token", return_value=token),
        patch("src.auth.get_authenticated_client", AsyncMock(return_value=mock_client)),
        patch("src.auth.close_client", AsyncMock()) as mock_close,
    ):
        # We need to manually iterate the async generator
        gen = get_authenticated_user(user_id=user_id, access_token=token)
        user = await anext(gen)

        assert isinstance(user, AuthenticatedUser)
        assert user.id == user_id
        assert user.client == mock_client

        # Verify close_client hasn't been called yet
        mock_close.assert_not_called()

        # Finish the generator
        try:
            await anext(gen)
        except StopAsyncIteration:
            pass

        # Verify close_client was called
        mock_close.assert_called_once_with(mock_client)
