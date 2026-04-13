"""Unit tests for the self-correcting RAG strategy."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.services.strategies.self_correcting import (
    MAX_RETRIES,
    MIN_RELEVANT_CHUNKS,
    RelevanceGrade,
    SelfCorrectingState,
)


def test_self_correcting_state_has_required_fields():
    """SelfCorrectingState has all required fields with correct defaults."""
    state = SelfCorrectingState(
        query="test question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=[],
        chunks=[],
        retry_count=0,
        answer="",
    )
    assert state["query"] == "test question"
    assert state["document_id"] == "doc-1"
    assert state["user_id"] == "user-1"
    assert state["k"] == 5
    assert state["retrieval_queries"] == []
    assert state["chunks"] == []
    assert state["retry_count"] == 0
    assert state["answer"] == ""


@pytest.mark.asyncio
async def test_retrieve_node_calls_retrieve_chunks():
    """retrieve_node calls retrieve_chunks and updates state."""
    from src.services.strategies.self_correcting import retrieve_node

    mock_chunks = [
        Document(page_content="chunk 1", metadata={"page": 1}),
    ]

    state = SelfCorrectingState(
        query="test question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=["test question"],
        chunks=[],
        retry_count=0,
        answer="",
    )

    with patch(
        "src.services.strategies.self_correcting.retrieve_chunks",
        new_callable=AsyncMock,
        return_value=mock_chunks,
    ) as mock_retrieve:
        result = await retrieve_node(state)

    mock_retrieve.assert_called_once_with(
        query="test question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
    )
    assert result["chunks"] == mock_chunks


@pytest.mark.asyncio
async def test_retrieve_node_uses_retrieval_query_when_set():
    """retrieve_node uses retrieval_query instead of original query when non-empty."""
    from src.services.strategies.self_correcting import retrieve_node

    state = SelfCorrectingState(
        query="original question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=["rewritten question"],
        chunks=[],
        retry_count=0,
        answer="",
    )

    with patch(
        "src.services.strategies.self_correcting.retrieve_chunks",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_retrieve:
        await retrieve_node(state)

    mock_retrieve.assert_called_once_with(
        query="rewritten question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
    )


@pytest.mark.asyncio
async def test_grade_relevance_node_marks_relevant_when_meets_threshold():
    """grade_relevance_node sets relevant=True and filters chunks when >= MIN_RELEVANT_CHUNKS pass."""
    from pydantic import SecretStr

    from src.services.strategies.self_correcting import grade_relevance_node

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"
    mock_settings.grading_llm_model = "gpt-4o-mini"

    relevant_doc_a = Document(page_content="relevant chunk A", metadata={"page": 1})
    relevant_doc_b = Document(page_content="relevant chunk B", metadata={"page": 2})
    relevant_doc_c = Document(page_content="relevant chunk C", metadata={"page": 3})
    irrelevant_doc = Document(page_content="noise", metadata={"page": 4})

    state = SelfCorrectingState(
        query="test question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=["test question"],
        chunks=[relevant_doc_a, relevant_doc_b, relevant_doc_c, irrelevant_doc],
        retry_count=0,
        answer="",
    )

    responses = [
        RelevanceGrade(relevant=True),
        RelevanceGrade(relevant=True),
        RelevanceGrade(relevant=True),
        RelevanceGrade(relevant=False),
    ]

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=responses)

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch(
            "src.services.strategies.self_correcting.get_settings",
            return_value=mock_settings,
        ),
        patch("src.services.strategies.self_correcting.init_chat_model"),
        patch(
            "src.services.strategies.self_correcting.pull_eval_prompt",
            new_callable=AsyncMock,
            return_value=mock_prompt,
        ),
    ):
        result = await grade_relevance_node(state)

    assert result["chunks"] == [relevant_doc_a, relevant_doc_b, relevant_doc_c]
    assert len(result["chunks"]) == MIN_RELEVANT_CHUNKS


@pytest.mark.asyncio
async def test_grade_relevance_node_marks_irrelevant_when_below_threshold():
    """grade_relevance_node returns empty chunks when fewer than MIN_RELEVANT_CHUNKS pass."""
    from pydantic import SecretStr

    from src.services.strategies.self_correcting import grade_relevance_node

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"
    mock_settings.grading_llm_model = "gpt-4o-mini"

    irrelevant_a = Document(page_content="irrelevant A", metadata={"page": 1})
    irrelevant_b = Document(page_content="irrelevant B", metadata={"page": 2})

    state = SelfCorrectingState(
        query="test question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=["test question"],
        chunks=[irrelevant_a, irrelevant_b],
        retry_count=0,
        answer="",
    )

    responses = [
        RelevanceGrade(relevant=False),
        RelevanceGrade(relevant=False),
    ]

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=responses)

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch(
            "src.services.strategies.self_correcting.get_settings",
            return_value=mock_settings,
        ),
        patch("src.services.strategies.self_correcting.init_chat_model"),
        patch(
            "src.services.strategies.self_correcting.pull_eval_prompt",
            new_callable=AsyncMock,
            return_value=mock_prompt,
        ),
    ):
        result = await grade_relevance_node(state)

    assert result["chunks"] == []


@pytest.mark.asyncio
async def test_grade_relevance_node_empty_chunks_marks_irrelevant():
    """grade_relevance_node returns empty chunks when there are no chunks to grade."""
    from src.services.strategies.self_correcting import grade_relevance_node

    state = SelfCorrectingState(
        query="test question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=["test question"],
        chunks=[],
        retry_count=0,
        answer="",
    )

    result = await grade_relevance_node(state)
    assert result["chunks"] == []


@pytest.mark.asyncio
async def test_grade_relevance_node_falls_back_on_exception():
    """grade_relevance_node clears chunks on LLM failure if retries remain."""
    from pydantic import SecretStr

    from src.services.strategies.self_correcting import grade_relevance_node

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"
    mock_settings.grading_llm_model = "gpt-4o-mini"

    state = SelfCorrectingState(
        query="test question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=["test question"],
        chunks=[
            Document(page_content="some chunk", metadata={"page": 1}),
        ],
        retry_count=0,
        answer="",
    )

    with (
        patch(
            "src.services.strategies.self_correcting.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.services.strategies.self_correcting.init_chat_model",
            side_effect=Exception("LLM down"),
        ),
    ):
        result = await grade_relevance_node(state)

    assert result["chunks"] == []


@pytest.mark.asyncio
async def test_grade_relevance_node_preserves_chunks_on_final_exception():
    """grade_relevance_node preserves existing chunks if LLM grading fails on the final attempt."""
    from pydantic import SecretStr

    from src.services.strategies.self_correcting import grade_relevance_node

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"
    mock_settings.grading_llm_model = "gpt-4o-mini"

    raw_chunks = [
        Document(page_content="some chunk", metadata={"page": 1}),
    ]
    state = SelfCorrectingState(
        query="test question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=["test question"],
        chunks=raw_chunks,
        retry_count=MAX_RETRIES,
        answer="",
    )

    with (
        patch(
            "src.services.strategies.self_correcting.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.services.strategies.self_correcting.init_chat_model",
            side_effect=Exception("LLM down"),
        ),
    ):
        result = await grade_relevance_node(state)

    # Should return empty dict so state is not updated (preserving chunks)
    assert result == {}


@pytest.mark.asyncio
async def test_rewrite_node_falls_back_on_exception():
    """rewrite_node returns original query when the LLM call fails."""
    from pydantic import SecretStr

    from src.services.strategies.self_correcting import rewrite_node

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    state = SelfCorrectingState(
        query="original question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=["original question"],
        chunks=[],
        retry_count=0,
        answer="",
    )

    with (
        patch(
            "src.services.strategies.self_correcting.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.services.strategies.self_correcting.init_chat_model",
            side_effect=Exception("LLM down"),
        ),
    ):
        result = await rewrite_node(state)

    assert result["retrieval_queries"] == ["original question"]
    assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_rewrite_node_rewrites_query_and_increments_retry():
    """rewrite_node rewrites the query and increments retry_count."""
    from pydantic import SecretStr

    from src.services.strategies.self_correcting import rewrite_node

    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    mock_response = MagicMock()
    mock_response.content = "improved search query"

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    state = SelfCorrectingState(
        query="original question",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
        retrieval_queries=["original question"],
        chunks=[],
        retry_count=0,
        answer="",
    )

    with (
        patch(
            "src.services.strategies.self_correcting.get_settings",
            return_value=mock_settings,
        ),
        patch("src.services.strategies.self_correcting.init_chat_model"),
        patch(
            "src.services.strategies.self_correcting.pull_eval_prompt",
            new_callable=AsyncMock,
            return_value=mock_prompt,
        ),
    ):
        result = await rewrite_node(state)

    assert result["retrieval_queries"] == ["original question", "improved search query"]
    assert result["retry_count"] == 1


def test_should_retry_returns_rewrite_when_below_threshold_and_can_retry():
    """should_retry returns 'rewrite' when chunk count is below MIN_RELEVANT_CHUNKS and retries remain."""
    from src.services.strategies.self_correcting import should_retry

    state = SelfCorrectingState(
        query="q",
        document_id="d",
        user_id="u",
        k=5,
        client=None,
        retrieval_queries=["q"],
        chunks=[],
        retry_count=0,
        answer="",
    )
    assert should_retry(state) == "rewrite"


def test_should_retry_returns_end_when_enough_chunks():
    """should_retry returns END when chunk count meets MIN_RELEVANT_CHUNKS."""
    from langgraph.graph import END

    from src.services.strategies.self_correcting import should_retry

    state = SelfCorrectingState(
        query="q",
        document_id="d",
        user_id="u",
        k=5,
        client=None,
        retrieval_queries=["q"],
        chunks=[
            Document(page_content="a", metadata={}),
            Document(page_content="b", metadata={}),
            Document(page_content="c", metadata={}),
        ],
        retry_count=0,
        answer="",
    )
    assert should_retry(state) == END


def test_should_retry_returns_end_when_max_retries_reached():
    """should_retry returns END when retry_count >= MAX_RETRIES."""
    from langgraph.graph import END

    from src.services.strategies.self_correcting import should_retry

    state = SelfCorrectingState(
        query="q",
        document_id="d",
        user_id="u",
        k=5,
        client=None,
        retrieval_queries=["q"],
        chunks=[],
        retry_count=3,
        answer="",
    )
    assert should_retry(state) == END


def test_build_graph_compiles_without_error():
    """build_graph returns a compiled graph with expected nodes."""
    from src.services.strategies.self_correcting import build_graph

    graph = build_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_execute_runs_graph_and_yields_stream_events():
    """execute runs the graph, then streams answer using the final chunks."""
    from src.schemas.chat import (
        ChunksEvent,
        Citation,
        CitationsEvent,
        RetrievedChunk,
        TokenEvent,
    )
    from src.services.strategies.self_correcting import execute

    final_chunks = [
        Document(page_content="final chunk", metadata={"page": 5}),
    ]

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "query": "original question",
            "chunks": final_chunks,
            "retry_count": 0,
        }
    )

    fake_events = [
        TokenEvent(token="Graph answer."),
        CitationsEvent(citations=[Citation(page=5)]),
    ]

    async def fake_stream(query, chunks):
        for event in fake_events:
            yield event

    with (
        patch(
            "src.services.strategies.self_correcting.build_graph",
            return_value=mock_graph,
        ),
        patch(
            "src.services.strategies.self_correcting.stream_answer_with_citations",
            side_effect=fake_stream,
        ) as mock_stream,
    ):
        results = [
            event
            async for event in execute(
                query="original question",  # ty: ignore[invalid-argument-type]
                document_id="doc-1",  # ty: ignore[invalid-argument-type]
                user_id="user-1",  # ty: ignore[invalid-argument-type]
                k=5,  # ty: ignore[invalid-argument-type]
            )
        ]

    mock_graph.ainvoke.assert_called_once()
    invoke_args = mock_graph.ainvoke.call_args[0][0]
    assert invoke_args["query"] == "original question"
    assert invoke_args["document_id"] == "doc-1"
    assert invoke_args["retrieval_queries"] == ["original question"]

    mock_stream.assert_called_once()
    call_kwargs = mock_stream.call_args
    assert call_kwargs[1]["query"] == "original question"

    assert len(results) == 3
    assert isinstance(results[0], ChunksEvent)
    assert results[0].chunks == [RetrievedChunk(page_content="final chunk", page=5)]
    assert results[1] == TokenEvent(token="Graph answer.")
    assert isinstance(results[2], CitationsEvent)
