from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr

from src.schemas.chat import (
    ChunksEvent,
    Citation,
    CitationsEvent,
    RetrievedChunk,
    TokenEvent,
)
from src.services.retrieval import (
    extract_citations,
    retrieve_chunks,
    stream_rag_pipeline,
)


@pytest.mark.asyncio
async def test_retrieve_chunks_calls_rpc_with_filter():
    """Test that retrieve_chunks calls RPC with correct filter."""
    mock_settings = MagicMock()
    mock_settings.embedding_model = "gemini-embedding-001"
    mock_settings.gemini_api_key = SecretStr("fake-key")
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
async def test_retrieve_chunks_uses_provided_client():
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
        patch("src.services.retrieval.get_settings", return_value=mock_settings),
        patch("src.services.retrieval.GoogleGenerativeAIEmbeddings") as mock_embed_cls,
        patch("src.services.retrieval.get_supabase_client") as mock_get_client,
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
async def test_stream_answer_with_citations_yields_tokens_then_citations():
    """Test streaming yields token strings then a citations tuple."""
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessageChunk

    from src.services.retrieval import stream_answer_with_citations

    chunks_input = [
        Document(page_content="content a", metadata={"page": 12}),
    ]

    async def fake_astream(prompt):
        for text in ["The author ", "argues that [p. 12]."]:
            yield AIMessageChunk(content=text)

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    with (
        patch("src.services.retrieval.get_settings", return_value=mock_settings),
        patch("src.services.retrieval.init_chat_model") as mock_init,
        patch("src.services.retrieval.pull_eval_prompt") as mock_pull_prompt,
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
async def test_stream_answer_empty_llm_response_yields_fallback_token():
    """Test that an empty LLM stream emits the fallback token then CitationsEvent."""
    from langchain_core.documents import Document

    from src.services.retrieval import stream_answer_with_citations

    chunks_input = [
        Document(page_content="content a", metadata={"page": 3}),
    ]

    async def empty_astream(prompt):
        return
        yield  # noqa: RET504 — makes this an async generator

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    with (
        patch("src.services.retrieval.get_settings", return_value=mock_settings),
        patch("src.services.retrieval.init_chat_model") as mock_init,
        patch("src.services.retrieval.pull_eval_prompt") as mock_pull_prompt,
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

    assert len(results) == 2
    assert results[0].type == "token"
    assert (
        results[0].token == "I don't have enough information to answer that question."
    )
    assert results[1].type == "citations"
    assert results[1].citations == []


@pytest.mark.asyncio
async def test_stream_answer_with_citations_no_chunks():
    """Test streaming with no chunks yields canned response and empty citations."""
    from src.services.retrieval import stream_answer_with_citations

    results = []
    async for item in stream_answer_with_citations(
        query="test question",  # ty: ignore[invalid-argument-type]
        chunks=[],  # ty: ignore[invalid-argument-type]
    ):
        results.append(item)

    assert results[0].type == "token"
    assert (
        results[0].token == "I don't have enough information to answer that question."
    )
    assert results[1].type == "citations"
    assert results[1].citations == []


@pytest.mark.asyncio
async def test_stream_rag_pipeline_yields_events():
    """stream_rag_pipeline retrieves chunks then yields stream events."""
    mock_chunks = [
        Document(page_content="chunk one", metadata={"page": 1}),
    ]

    fake_events = [
        TokenEvent(token="Hello "),
        TokenEvent(token="world."),
        CitationsEvent(citations=[Citation(page=1)]),
    ]

    async def fake_stream(query: str, chunks: list[Document]):
        for event in fake_events:
            yield event

    with (
        patch(
            "src.services.retrieval.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=mock_chunks,
        ) as mock_retrieve,
        patch(
            "src.services.retrieval.stream_answer_with_citations",
            side_effect=fake_stream,
        ),
    ):
        mock_client = AsyncMock()
        results = [
            event
            async for event in stream_rag_pipeline(
                query="Hi",  # ty: ignore[invalid-argument-type]
                document_id="doc-1",  # ty: ignore[invalid-argument-type]
                user_id="user-1",  # ty: ignore[invalid-argument-type]
                k=3,  # ty: ignore[invalid-argument-type]
                client=mock_client,
            )
        ]

    assert len(results) == 4

    # First event is ChunksEvent with retrieved documents
    assert isinstance(results[0], ChunksEvent)
    assert results[0].chunks == [
        RetrievedChunk(page_content="chunk one", page=1),
    ]

    # Then token events and citations
    assert results[1] == TokenEvent(token="Hello ")
    assert results[2] == TokenEvent(token="world.")
    assert isinstance(results[3], CitationsEvent)
    assert results[3].citations == [Citation(page=1)]

    # Verify retrieve_chunks was called with forwarded args
    mock_retrieve.assert_called_once_with(
        query="Hi",
        document_id="doc-1",
        user_id="user-1",
        k=3,
        client=mock_client,
    )
