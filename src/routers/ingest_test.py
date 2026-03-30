"""Tests for the ingest router endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from src.app import app
from src.auth import get_authenticated_user
from src.services.exceptions import DocumentNotFoundError, DownloadError

USER_ID = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"
JWT_SECRET = "test-jwt-secret-for-auth-tests-minimum-32-bytes"


@pytest.fixture(autouse=True)
def _patch_jwt_secret():
    """Patch get_settings so the auth dependency uses our known secret."""
    mock_settings = MagicMock()
    mock_settings.supabase_jwt_secret = SecretStr(JWT_SECRET)
    with patch("src.auth.get_settings", return_value=mock_settings):
        yield


def _make_token(user_id: str = USER_ID) -> str:
    return pyjwt.encode(
        {"sub": user_id, "aud": "authenticated"},
        JWT_SECRET,
        algorithm="HS256",
    )


@pytest.mark.asyncio
async def test_ingest_document_success():
    """Test successful document ingestion pipeline."""
    document_id = "test-doc-123"

    mock_doc = {
        "id": document_id,
        "user_id": USER_ID,
        "blob_url": "https://example.com/test.pdf",
        "ingested_at": None,
    }
    mock_pdf_bytes = b"%PDF-1.4 fake content"
    mock_chunks = [
        MagicMock(page_content="Chunk 1", metadata={"page": 0}),
        MagicMock(page_content="Chunk 2", metadata={"page": 0}),
    ]

    mock_auth_user = MagicMock()
    mock_auth_user.id = USER_ID
    mock_auth_user.client = AsyncMock()

    app.dependency_overrides[get_authenticated_user] = lambda: mock_auth_user

    try:
        with (
            patch(
                "src.routers.ingest.get_document", new_callable=AsyncMock
            ) as mock_get,
            patch(
                "src.routers.ingest.download_pdf", new_callable=AsyncMock
            ) as mock_download,
            patch(
                "src.routers.ingest.chunk_pdf", return_value=mock_chunks
            ) as mock_chunk,
            patch(
                "src.routers.ingest.embed_and_store", new_callable=AsyncMock
            ) as mock_embed,
            patch(
                "src.routers.ingest.mark_ingested", new_callable=AsyncMock
            ) as mock_mark,
        ):
            mock_get.return_value = mock_doc
            mock_download.return_value = mock_pdf_bytes

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    f"/ingest/{document_id}",
                    headers={"Authorization": f"Bearer {_make_token()}"},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ingested"
            assert data["chunk_count"] == 2

            mock_get.assert_called_once_with(
                document_id, USER_ID, client=mock_auth_user.client
            )
            mock_download.assert_called_once_with(mock_doc["blob_url"])
            mock_chunk.assert_called_once_with(mock_pdf_bytes)
            mock_embed.assert_called_once_with(
                mock_chunks, document_id, USER_ID, client=mock_auth_user.client
            )
            mock_mark.assert_called_once_with(document_id, client=mock_auth_user.client)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_document_already_ingested():
    """Test ingestion skips when document already ingested."""
    document_id = "test-doc-456"

    mock_doc = {
        "id": document_id,
        "user_id": USER_ID,
        "blob_url": "https://example.com/test.pdf",
        "ingested_at": "2026-03-27T12:00:00Z",
    }

    mock_auth_user = MagicMock()
    mock_auth_user.id = USER_ID
    mock_auth_user.client = AsyncMock()

    app.dependency_overrides[get_authenticated_user] = lambda: mock_auth_user

    try:
        with (
            patch(
                "src.routers.ingest.get_document", new_callable=AsyncMock
            ) as mock_get,
        ):
            mock_get.return_value = mock_doc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    f"/ingest/{document_id}",
                    headers={"Authorization": f"Bearer {_make_token()}"},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "already_ingested"

            mock_get.assert_called_once_with(
                document_id, USER_ID, client=mock_auth_user.client
            )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_document_not_found():
    """Test ingestion handles document not found error."""
    document_id = "nonexistent-doc"

    mock_auth_user = MagicMock()
    mock_auth_user.id = USER_ID
    mock_auth_user.client = AsyncMock()

    app.dependency_overrides[get_authenticated_user] = lambda: mock_auth_user

    try:
        with (
            patch(
                "src.routers.ingest.get_document", new_callable=AsyncMock
            ) as mock_get,
        ):
            mock_get.side_effect = DocumentNotFoundError(document_id)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    f"/ingest/{document_id}",
                    headers={"Authorization": f"Bearer {_make_token()}"},
                )

            assert response.status_code == 404
            assert "Document not found" in response.json()["detail"]
            mock_get.assert_called_once_with(
                document_id, USER_ID, client=mock_auth_user.client
            )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_document_download_error():
    """Test ingestion handles download error."""
    document_id = "test-doc-789"

    mock_doc = {
        "id": document_id,
        "user_id": USER_ID,
        "blob_url": "https://example.com/test.pdf",
        "ingested_at": None,
    }

    mock_auth_user = MagicMock()
    mock_auth_user.id = USER_ID
    mock_auth_user.client = AsyncMock()

    app.dependency_overrides[get_authenticated_user] = lambda: mock_auth_user

    try:
        with (
            patch(
                "src.routers.ingest.get_document", new_callable=AsyncMock
            ) as mock_get,
            patch(
                "src.routers.ingest.download_pdf", new_callable=AsyncMock
            ) as mock_download,
        ):
            mock_get.return_value = mock_doc
            mock_download.side_effect = DownloadError(mock_doc["blob_url"], 403)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    f"/ingest/{document_id}",
                    headers={"Authorization": f"Bearer {_make_token()}"},
                )

            assert response.status_code == 502
            assert "Failed to download" in response.json()["detail"]
            mock_get.assert_called_once_with(
                document_id, USER_ID, client=mock_auth_user.client
            )
            mock_download.assert_called_once_with(mock_doc["blob_url"])
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_document_with_zero_chunks():
    """Test ingestion handles PDF that produces no chunks."""
    document_id = "test-doc-empty"

    mock_doc = {
        "id": document_id,
        "user_id": USER_ID,
        "blob_url": "https://example.com/test.pdf",
        "ingested_at": None,
    }
    mock_pdf_bytes = b"%PDF-1.4 blank content"
    mock_chunks = []

    mock_auth_user = MagicMock()
    mock_auth_user.id = USER_ID
    mock_auth_user.client = AsyncMock()

    app.dependency_overrides[get_authenticated_user] = lambda: mock_auth_user

    try:
        with (
            patch(
                "src.routers.ingest.get_document", new_callable=AsyncMock
            ) as mock_get,
            patch(
                "src.routers.ingest.download_pdf", new_callable=AsyncMock
            ) as mock_download,
            patch(
                "src.routers.ingest.chunk_pdf", return_value=mock_chunks
            ) as mock_chunk,
            patch(
                "src.routers.ingest.embed_and_store", new_callable=AsyncMock
            ) as mock_embed,
            patch(
                "src.routers.ingest.mark_ingested", new_callable=AsyncMock
            ) as mock_mark,
        ):
            mock_get.return_value = mock_doc
            mock_download.return_value = mock_pdf_bytes

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    f"/ingest/{document_id}",
                    headers={"Authorization": f"Bearer {_make_token()}"},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ingested"
            assert data["chunk_count"] == 0

            mock_get.assert_called_once_with(
                document_id, USER_ID, client=mock_auth_user.client
            )
            mock_download.assert_called_once_with(mock_doc["blob_url"])
            mock_chunk.assert_called_once_with(mock_pdf_bytes)
            mock_embed.assert_called_once_with(
                mock_chunks, document_id, USER_ID, client=mock_auth_user.client
            )
            mock_mark.assert_called_once_with(document_id, client=mock_auth_user.client)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_without_auth_returns_401():
    """Ingest endpoint without auth header returns 401."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/ingest/some-doc-id")
    assert response.status_code == 401
