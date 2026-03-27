from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.schemas.chat import Citation


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_chat_endpoint_returns_answer_with_citations(client):
    """Test that chat endpoint returns answer with citations."""
    document_id = "test-doc-123"

    with (
        patch("src.routers.chat.retrieve_chunks") as mock_retrieve,
        patch("src.routers.chat.generate_answer_with_citations") as mock_generate,
    ):
        mock_retrieve.return_value = []  # Return empty chunks for simplicity
        mock_generate.return_value = (
            "The author argues this point [p. 12].",
            [Citation(page=12)],
        )

        response = client.post(
            f"/chat/{document_id}",
            json={"message": "What is the main argument?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert len(data["citations"]) == 1
        assert data["citations"][0]["page"] == 12


@pytest.mark.asyncio
async def test_chat_endpoint_with_no_results(client):
    """Test chat endpoint when no relevant chunks found."""
    document_id = "test-doc-456"

    with (
        patch("src.routers.chat.retrieve_chunks") as mock_retrieve,
        patch("src.routers.chat.generate_answer_with_citations") as mock_generate,
    ):
        mock_retrieve.return_value = []
        mock_generate.return_value = (
            "I don't have enough information.",
            [],
        )

        response = client.post(
            f"/chat/{document_id}",
            json={"message": "Unknown topic"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["citations"] == []


@pytest.mark.asyncio
async def test_chat_endpoint_error_handling(client):
    """Test chat endpoint error handling."""
    document_id = "test-doc-789"

    with patch("src.routers.chat.retrieve_chunks") as mock_retrieve:
        mock_retrieve.side_effect = Exception("Database error")

        response = client.post(
            f"/chat/{document_id}",
            json={"message": "Test question"},
        )

        assert response.status_code == 500
