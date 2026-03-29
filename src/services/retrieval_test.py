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
    mock_settings = MagicMock()
    mock_settings.embedding_model = "gemini-embedding-001"
    mock_settings.gemini_api_key = "fake-key"
    mock_settings.min_similarity = 0.5

    with (
        patch("src.services.retrieval.get_settings", return_value=mock_settings),
        patch("src.services.retrieval.GoogleGenerativeAIEmbeddings") as mock_embed_cls,
        patch("src.services.retrieval.get_supabase_client") as mock_client_factory,
    ):
        mock_embed_instance = MagicMock()
        mock_embed_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock_embed_cls.return_value = mock_embed_instance
        mock_client = AsyncMock()
        mock_client_factory.return_value = mock_client

        # Mock RPC result - rpc().execute() is the chain
        mock_result = MagicMock()
        mock_result.data = [
            {"content": "Test content", "metadata": {"page": 5}, "similarity": 0.8},
            {"content": "More content", "metadata": {"page": 10}, "similarity": 0.7},
            {"content": "Low similarity", "metadata": {"page": 1}, "similarity": 0.1},
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
        assert len(result) == 2  # The 0.1 similarity entry should be filtered out
        assert result[0].page_content == "Test content"
        assert result[0].metadata["page"] == 5


def test_extract_citations_returns_only_cited_pages():
    """Test citation extraction returns only pages cited in the answer."""
    from langchain_core.documents import Document

    chunks = [
        Document(page_content="content a", metadata={"page": 15}),
        Document(page_content="content b", metadata={"page": 12}),
        Document(page_content="content c", metadata={"page": 15}),
    ]
    answer = "Some info [p. 12] and more [p. 15]."
    citations = extract_citations(answer, chunks)

    assert len(citations) == 2
    assert citations[0].page == 12
    assert citations[1].page == 15


def test_extract_citations_ignores_pages_not_in_chunks():
    """Test that cited pages not present in chunks are excluded."""
    from langchain_core.documents import Document

    chunks = [
        Document(page_content="content a", metadata={"page": 5}),
    ]
    answer = "See [p. 5] and [p. 99]."
    citations = extract_citations(answer, chunks)

    assert len(citations) == 1
    assert citations[0].page == 5


def test_extract_citations_no_markers_returns_empty():
    """Test that an answer with no page markers yields no citations."""
    from langchain_core.documents import Document

    chunks = [
        Document(page_content="content a", metadata={"page": 5}),
    ]
    citations = extract_citations("No page markers here.", chunks)
    assert citations == []


def test_extract_citations_empty():
    """Test citation extraction with no chunks."""
    citations = extract_citations("Some answer [p. 1].", [])
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
