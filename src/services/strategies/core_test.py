from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from pydantic import SecretStr

from src.schemas.chat import Citation
from src.services.strategies.core import (
    extract_citations,
    fuse_rrf,
    retrieve_chunks,
    stream_answer_with_citations,
)


@pytest.mark.asyncio
async def test_retrieve_chunks_calls_rpc_with_filter() -> None:
    """Test that retrieve_chunks calls RPC with correct filter."""
    mock_settings = MagicMock()
    mock_settings.embedding_model = "gemini-embedding-001"
    mock_settings.gemini_api_key = SecretStr("fake-key")
    mock_settings.min_similarity = 0.5

    with (
        patch("src.services.strategies.core.get_settings", return_value=mock_settings),
        patch(
            "src.services.strategies.core.GoogleGenerativeAIEmbeddings"
        ) as mock_embed_cls,
        patch(
            "src.services.strategies.core.get_supabase_client"
        ) as mock_client_factory,
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
            query="test question",  # ty: ignore[invalid-argument-type]
            document_id="doc-123",  # ty: ignore[invalid-argument-type]
            user_id="user-456",  # ty: ignore[invalid-argument-type]
            k=3,  # ty: ignore[invalid-argument-type]
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


@pytest.mark.asyncio
async def test_retrieve_chunks_uses_provided_client() -> None:
    """retrieve_chunks uses the explicitly provided client instead of the default."""
    mock_settings = MagicMock()
    mock_settings.embedding_model = "gemini-embedding-001"
    mock_settings.gemini_api_key = SecretStr("fake-key")
    mock_settings.min_similarity = 0.5

    mock_client = AsyncMock()
    mock_result = MagicMock()
    mock_result.data = []

    # Mock RPC execution
    mock_execute = AsyncMock(return_value=mock_result)
    mock_rpc_builder = MagicMock()
    mock_rpc_builder.execute = mock_execute
    mock_client.rpc = MagicMock(return_value=mock_rpc_builder)

    with (
        patch("src.services.strategies.core.get_settings", return_value=mock_settings),
        patch(
            "src.services.strategies.core.GoogleGenerativeAIEmbeddings"
        ) as mock_embed_cls,
        patch("src.services.strategies.core.get_supabase_client") as mock_get_client,
    ):
        mock_embed_instance = MagicMock()
        mock_embed_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock_embed_cls.return_value = mock_embed_instance

        result = await retrieve_chunks(
            query="test",  # ty: ignore[invalid-argument-type]
            document_id="doc-123",  # ty: ignore[invalid-argument-type]
            user_id="user-456",  # ty: ignore[invalid-argument-type]
            client=mock_client,
        )

        assert result == []
        mock_client.rpc.assert_called_once()
        mock_get_client.assert_not_called()


def test_extract_citations_returns_only_cited_pages() -> None:
    """Test citation extraction returns only pages cited in the answer."""
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


def test_extract_citations_ignores_pages_not_in_chunks() -> None:
    """Test that cited pages not present in chunks are excluded."""
    chunks = [
        Document(page_content="content a", metadata={"page": 5}),
    ]
    answer = "See [p. 5] and [p. 99]."
    citations = extract_citations(answer, chunks)

    assert len(citations) == 1
    assert citations[0].page == 5


def test_extract_citations_no_markers_returns_empty() -> None:
    """Test that an answer with no page markers yields no citations."""
    chunks = [
        Document(page_content="content a", metadata={"page": 5}),
    ]
    citations = extract_citations("No page markers here.", chunks)
    assert citations == []


def test_extract_citations_empty() -> None:
    """Test citation extraction with no chunks."""
    citations = extract_citations("Some answer [p. 1].", [])
    assert citations == []


@pytest.mark.asyncio
async def test_stream_answer_with_citations_yields_tokens_then_citations() -> None:
    """Test streaming yields token strings then a citations tuple."""
    chunks_input = [
        Document(page_content="content a", metadata={"page": 12}),
    ]

    async def fake_astream(prompt: Any) -> AsyncIterator[AIMessageChunk]:
        """Yield fake LLM message chunks for testing."""
        for text in ["The author ", "argues that [p. 12]."]:
            yield AIMessageChunk(content=text)

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    with (
        patch("src.services.strategies.core.get_settings", return_value=mock_settings),
        patch("src.services.strategies.core.init_chat_model") as mock_init,
        patch(
            "src.services.strategies.core.pull_eval_prompt",
            new_callable=AsyncMock,
        ) as mock_pull_prompt,
    ):
        # Mock the chain's astream by mocking the prompt and the resulting chain
        mock_chain = MagicMock()
        mock_chain.astream = fake_astream

        mock_prompt_template = MagicMock()
        mock_prompt_template.__or__ = MagicMock(return_value=mock_chain)
        mock_pull_prompt.return_value = mock_prompt_template

        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        results = []
        async for item in stream_answer_with_citations(
            query="What is the main argument?",  # ty: ignore[invalid-argument-type]
            chunks=chunks_input,  # ty: ignore[invalid-argument-type]
        ):
            results.append(item)

    mock_pull_prompt.assert_awaited_once_with("sanctuary-rag-prompt")

    # All items except the last are TokenEvents
    tokens = results[:-1]
    assert all(e.type == "token" for e in tokens)
    assert [e.token for e in tokens] == ["The author ", "argues that [p. 12]."]

    # Last item is a CitationsEvent
    final = results[-1]
    assert final.type == "citations"
    assert len(final.citations) == 1
    assert final.citations[0].page == 12


@pytest.mark.asyncio
async def test_stream_answer_empty_llm_response_yields_fallback_token() -> None:
    """Test that an empty LLM stream emits the fallback token then CitationsEvent."""
    chunks_input = [
        Document(page_content="content a", metadata={"page": 3}),
    ]

    async def empty_astream(prompt: Any) -> AsyncIterator[AIMessageChunk]:
        """Yield nothing, simulating an empty LLM response."""
        if False:
            yield AIMessageChunk(content="")

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    with (
        patch("src.services.strategies.core.get_settings", return_value=mock_settings),
        patch("src.services.strategies.core.init_chat_model") as mock_init,
        patch(
            "src.services.strategies.core.pull_eval_prompt",
            new_callable=AsyncMock,
        ) as mock_pull_prompt,
    ):
        # Mock the chain's astream by mocking the prompt and the resulting chain
        mock_chain = MagicMock()
        mock_chain.astream = empty_astream

        mock_prompt_template = MagicMock()
        mock_prompt_template.__or__ = MagicMock(return_value=mock_chain)
        mock_pull_prompt.return_value = mock_prompt_template

        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        results = []
        async for item in stream_answer_with_citations(
            query="anything",  # ty: ignore[invalid-argument-type]
            chunks=chunks_input,  # ty: ignore[invalid-argument-type]
        ):
            results.append(item)

    mock_pull_prompt.assert_awaited_once_with("sanctuary-rag-prompt")

    assert len(results) == 2
    assert results[0].type == "token"
    assert (
        results[0].token
        == "Something went wrong while generating the answer. Please try again."
    )
    assert results[1].type == "citations"
    assert results[1].citations == []


@pytest.mark.asyncio
async def test_stream_answer_with_citations_no_chunks() -> None:
    """Test streaming with no chunks yields canned response and empty citations."""
    results = []
    async for item in stream_answer_with_citations(
        query="test question",  # ty: ignore[invalid-argument-type]
        chunks=[],  # ty: ignore[invalid-argument-type]
    ):
        results.append(item)

    assert results[0].type == "token"
    assert (
        results[0].token
        == "The document doesn't contain enough information to answer this question. Try rephrasing or checking other documents."
    )
    assert results[1].type == "citations"
    assert results[1].citations == []


def test_fuse_rrf_single_list_preserves_order() -> None:
    """A single ranked list is returned in its original order."""
    docs = [
        Document(page_content="first", metadata={"page": 1}),
        Document(page_content="second", metadata={"page": 2}),
        Document(page_content="third", metadata={"page": 3}),
    ]
    result = fuse_rrf([docs])
    assert [d.page_content for d in result] == ["first", "second", "third"]


def test_fuse_rrf_overlapping_lists_ranks_by_fusion_score() -> None:
    """Documents appearing in more lists rank higher than single-list documents."""
    list_a = [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 2}),
    ]
    list_b = [
        Document(page_content="B", metadata={"page": 2}),
        Document(page_content="C", metadata={"page": 3}),
    ]
    result = fuse_rrf([list_a, list_b])
    contents = [d.page_content for d in result]
    # B appears in both lists (rank 2 + rank 1) -> highest score
    # A appears at rank 1 in list_a only
    # C appears at rank 2 in list_b only -> same score as A, but A seen first
    assert contents == ["B", "A", "C"]


def test_fuse_rrf_no_overlap_orders_by_rank() -> None:
    """Non-overlapping lists: all documents included, ordered by individual RRF score."""
    list_a = [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 2}),
    ]
    list_b = [
        Document(page_content="C", metadata={"page": 3}),
        Document(page_content="D", metadata={"page": 4}),
    ]
    result = fuse_rrf([list_a, list_b])
    contents = [d.page_content for d in result]
    # A and C both at rank 1 -> same score, A seen first
    # B and D both at rank 2 -> same score, B seen first
    assert contents == ["A", "C", "B", "D"]


def test_fuse_rrf_empty_input_returns_empty() -> None:
    """Empty input returns empty list."""
    assert fuse_rrf([]) == []


def test_fuse_rrf_empty_sublists_returns_empty() -> None:
    """All-empty sublists returns empty list."""
    assert fuse_rrf([[], []]) == []


def test_fuse_rrf_max_chunks_limits_output() -> None:
    """max_chunks truncates the result to the top-N documents."""
    list_a = [
        Document(page_content="A", metadata={"page": 1}),
        Document(page_content="B", metadata={"page": 2}),
        Document(page_content="C", metadata={"page": 3}),
    ]
    list_b = [
        Document(page_content="B", metadata={"page": 2}),
        Document(page_content="D", metadata={"page": 4}),
    ]
    result = fuse_rrf([list_a, list_b], max_chunks=2)
    assert len(result) == 2
    # B has highest score (in both lists), A is next (rank 1 in list_a)
    assert result[0].page_content == "B"
    assert result[1].page_content == "A"


def test_fuse_rrf_preserves_metadata() -> None:
    """Metadata from the first occurrence of a document is preserved."""
    list_a = [Document(page_content="A", metadata={"page": 1, "source": "a"})]
    list_b = [Document(page_content="A", metadata={"page": 1, "source": "b"})]
    result = fuse_rrf([list_a, list_b])
    assert len(result) == 1
    assert result[0].metadata["source"] == "a"
