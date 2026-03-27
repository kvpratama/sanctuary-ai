"""Tests for the ingest router endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.services.exceptions import DocumentNotFoundError, DownloadError


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_ingest_document_success(client):
    """Test successful document ingestion pipeline."""
    document_id = "test-doc-123"
    user_id = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"

    mock_doc = {
        "id": document_id,
        "user_id": user_id,
        "blob_url": "https://example.com/test.pdf",
        "ingested_at": None,
    }
    mock_pdf_bytes = b"%PDF-1.4 fake content"
    mock_chunks = [
        AsyncMock(page_content="Chunk 1", metadata={"page": 0}),
        AsyncMock(page_content="Chunk 2", metadata={"page": 0}),
    ]

    with (
        patch("src.routers.ingest.get_document", new_callable=AsyncMock) as mock_get,
        patch(
            "src.routers.ingest.download_pdf", new_callable=AsyncMock
        ) as mock_download,
        patch("src.routers.ingest.chunk_pdf", return_value=mock_chunks) as mock_chunk,
        patch(
            "src.routers.ingest.embed_and_store", new_callable=AsyncMock
        ) as mock_embed,
        patch("src.routers.ingest.mark_ingested", new_callable=AsyncMock) as mock_mark,
    ):
        mock_get.return_value = mock_doc
        mock_download.return_value = mock_pdf_bytes

        response = client.post(f"/ingest/{document_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ingested"
        assert data["chunk_count"] == 2

        mock_get.assert_called_once_with(document_id, user_id)
        mock_download.assert_called_once_with(mock_doc["blob_url"])
        mock_chunk.assert_called_once_with(mock_pdf_bytes)
        mock_embed.assert_called_once_with(mock_chunks, document_id, user_id)
        mock_mark.assert_called_once_with(document_id)


@pytest.mark.asyncio
async def test_ingest_document_already_ingested(client):
    """Test ingestion skips when document already ingested."""
    document_id = "test-doc-456"
    user_id = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"

    mock_doc = {
        "id": document_id,
        "user_id": user_id,
        "blob_url": "https://example.com/test.pdf",
        "ingested_at": "2026-03-27T12:00:00Z",
    }

    with patch("src.routers.ingest.get_document", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_doc

        response = client.post(f"/ingest/{document_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_ingested"

        mock_get.assert_called_once_with(document_id, user_id)


@pytest.mark.asyncio
async def test_ingest_document_not_found(client):
    """Test ingestion handles document not found error."""
    document_id = "nonexistent-doc"
    user_id = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"

    with patch("src.routers.ingest.get_document", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = DocumentNotFoundError(document_id)

        with pytest.raises(DocumentNotFoundError):
            client.post(f"/ingest/{document_id}")

        mock_get.assert_called_once_with(document_id, user_id)


@pytest.mark.asyncio
async def test_ingest_document_download_error(client):
    """Test ingestion handles download error."""
    document_id = "test-doc-789"
    user_id = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"

    mock_doc = {
        "id": document_id,
        "user_id": user_id,
        "blob_url": "https://example.com/test.pdf",
        "ingested_at": None,
    }

    with (
        patch("src.routers.ingest.get_document", new_callable=AsyncMock) as mock_get,
        patch(
            "src.routers.ingest.download_pdf", new_callable=AsyncMock
        ) as mock_download,
    ):
        mock_get.return_value = mock_doc
        mock_download.side_effect = DownloadError(mock_doc["blob_url"], 403)

        with pytest.raises(DownloadError):
            client.post(f"/ingest/{document_id}")

        mock_get.assert_called_once_with(document_id, user_id)
        mock_download.assert_called_once_with(mock_doc["blob_url"])


@pytest.mark.asyncio
async def test_ingest_document_with_zero_chunks(client):
    """Test ingestion handles PDF that produces no chunks."""
    document_id = "test-doc-empty"
    user_id = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"

    mock_doc = {
        "id": document_id,
        "user_id": user_id,
        "blob_url": "https://example.com/test.pdf",
        "ingested_at": None,
    }
    mock_pdf_bytes = b"%PDF-1.4 blank content"
    mock_chunks = []

    with (
        patch("src.routers.ingest.get_document", new_callable=AsyncMock) as mock_get,
        patch(
            "src.routers.ingest.download_pdf", new_callable=AsyncMock
        ) as mock_download,
        patch("src.routers.ingest.chunk_pdf", return_value=mock_chunks) as mock_chunk,
        patch(
            "src.routers.ingest.embed_and_store", new_callable=AsyncMock
        ) as mock_embed,
        patch("src.routers.ingest.mark_ingested", new_callable=AsyncMock) as mock_mark,
    ):
        mock_get.return_value = mock_doc
        mock_download.return_value = mock_pdf_bytes

        response = client.post(f"/ingest/{document_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ingested"
        assert data["chunk_count"] == 0

        mock_get.assert_called_once_with(document_id, user_id)
        mock_download.assert_called_once_with(mock_doc["blob_url"])
        mock_chunk.assert_called_once_with(mock_pdf_bytes)
        mock_embed.assert_called_once_with(mock_chunks, document_id, user_id)
        mock_mark.assert_called_once_with(document_id)
