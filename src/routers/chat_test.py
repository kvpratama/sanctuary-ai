from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.app import app
from src.schemas.chat import Citation

client = TestClient(app)


def test_chat_endpoint_returns_answer_with_citations():
    """Test that chat endpoint returns answer with citations."""
    document_id = "test-doc-123"

    with (
        patch(
            "src.routers.chat.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.routers.chat.generate_answer_with_citations",
            new_callable=AsyncMock,
            return_value=(
                "The author argues this point [p. 12].",
                [Citation(page=12)],
            ),
        ),
    ):
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


def test_chat_endpoint_with_no_results():
    """Test chat endpoint when no relevant chunks found."""
    document_id = "test-doc-456"

    with (
        patch(
            "src.routers.chat.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.routers.chat.generate_answer_with_citations",
            new_callable=AsyncMock,
            return_value=("I don't have enough information.", []),
        ),
    ):
        response = client.post(
            f"/chat/{document_id}",
            json={"message": "Unknown topic"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["citations"] == []


def test_chat_endpoint_error_handling():
    """Test chat endpoint error handling."""
    document_id = "test-doc-789"

    with patch(
        "src.routers.chat.retrieve_chunks",
        new_callable=AsyncMock,
        side_effect=Exception("Database error"),
    ):
        response = client.post(
            f"/chat/{document_id}",
            json={"message": "Test question"},
        )

        assert response.status_code == 500
