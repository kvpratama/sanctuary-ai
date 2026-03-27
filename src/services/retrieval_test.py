from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.retrieval import (
    extract_citations,
    generate_answer_with_citations,
    retrieve_chunks,
)


@pytest.mark.asyncio
async def test_retrieve_chunks_calls_rpc_with_filter():
    """Test that retrieve_chunks calls RPC with correct filter."""
    with patch("src.services.retrieval.get_supabase_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client_factory.return_value = mock_client

        # Mock RPC result - rpc().execute() is the chain
        mock_result = MagicMock()
        mock_result.data = [
            {"content": "Test content", "metadata": {"page": 5}},
            {"content": "More content", "metadata": {"page": 10}},
        ]

        # Create a mock for the RPC call chain
        # rpc() returns a builder (not awaitable), execute() is awaitable
        mock_execute = AsyncMock(return_value=mock_result)
        mock_rpc_builder = MagicMock()
        mock_rpc_builder.execute = mock_execute
        mock_client.rpc = MagicMock(return_value=mock_rpc_builder)

        result = await retrieve_chunks(
            query="test question",
            document_id="doc-123",
            user_id="user-456",
            k=3,
        )

        # Verify RPC was called with correct parameters
        mock_client.rpc.assert_called_once()
        call_args = mock_client.rpc.call_args
        assert call_args[0][0] == "match_document_embeddings"
        params = call_args[0][1]
        assert params["filter"]["document_id"] == "doc-123"
        assert params["filter"]["user_id"] == "user-456"
        assert params["match_count"] == 3

        # Verify result conversion
        assert len(result) == 2
        assert result[0].page_content == "Test content"
        assert result[0].metadata["page"] == 5


def test_extract_citations_finds_page_numbers():
    """Test citation extraction from LLM answer."""
    answer = (
        "The author argues this point [p. 12]. Furthermore, on page 15, it states..."
    )
    citations = extract_citations(answer)

    assert len(citations) == 2
    assert citations[0].page == 12
    assert citations[1].page == 15


def test_extract_citations_empty():
    """Test citation extraction with no citations."""
    answer = "No citations here."
    citations = extract_citations(answer)
    assert citations == []


@pytest.mark.asyncio
async def test_generate_answer_with_no_chunks():
    """Test generate_answer_with_citations with no chunks."""
    answer, citations = await generate_answer_with_citations(
        query="test question",
        chunks=[],
    )

    assert "don't have enough information" in answer.lower()
    assert citations == []
