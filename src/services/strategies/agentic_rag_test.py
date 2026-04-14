"""Tests for the agentic RAG strategy."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from src.schemas.chat import ChunksEvent, CitationsEvent, TokenEvent

# --- _make_search_tool tests ---


@pytest.mark.asyncio
async def test_search_tool_returns_chunk_content():
    """The search_docs tool should return formatted chunk content."""
    from src.services.strategies.agentic_rag import _make_search_tool

    mock_chunks = [
        Document(page_content="The sky is blue.", metadata={"page": 1}),
        Document(page_content="Clouds are white.", metadata={"page": 1}),
    ]

    accumulated: list[Document] = []

    with patch(
        "src.services.strategies.agentic_rag.retrieve_chunks",
        new_callable=AsyncMock,
        return_value=mock_chunks,
    ):
        search_tool = _make_search_tool(
            document_id="doc-123",
            user_id="user-456",
            k=5,
            client=None,
            accumulated_chunks=accumulated,
        )
        result = await search_tool.ainvoke({"query": "sky color"})

    assert "The sky is blue." in result
    assert "Clouds are white." in result


@pytest.mark.asyncio
async def test_search_tool_accumulates_new_chunks():
    """The search_docs tool should deduplicate and accumulate new chunks."""
    from src.services.strategies.agentic_rag import _make_search_tool

    existing = Document(page_content="Existing content.", metadata={"page": 1})
    new_chunk = Document(page_content="New content.", metadata={"page": 2})
    accumulated: list[Document] = [existing]

    with patch(
        "src.services.strategies.agentic_rag.retrieve_chunks",
        new_callable=AsyncMock,
        return_value=[existing, new_chunk],
    ):
        search_tool = _make_search_tool(
            document_id="doc-123",
            user_id="user-456",
            k=5,
            client=None,
            accumulated_chunks=accumulated,
        )
        await search_tool.ainvoke({"query": "test query"})

    assert len(accumulated) == 2


@pytest.mark.asyncio
async def test_search_tool_returns_no_results_message():
    """The search_docs tool should return a message when no chunks found."""
    from src.services.strategies.agentic_rag import _make_search_tool

    accumulated: list[Document] = []

    with patch(
        "src.services.strategies.agentic_rag.retrieve_chunks",
        new_callable=AsyncMock,
        return_value=[],
    ):
        search_tool = _make_search_tool(
            document_id="doc-123",
            user_id="user-456",
            k=5,
            client=None,
            accumulated_chunks=accumulated,
        )
        result = await search_tool.ainvoke({"query": "nothing found"})

    assert "No relevant information found" in result


@pytest.mark.asyncio
async def test_search_tool_returns_no_additional_message():
    """The search_docs tool should return message when all chunks are duplicates."""
    from src.services.strategies.agentic_rag import _make_search_tool

    existing = Document(page_content="Already have this.", metadata={"page": 1})
    accumulated: list[Document] = [existing]

    with patch(
        "src.services.strategies.agentic_rag.retrieve_chunks",
        new_callable=AsyncMock,
        return_value=[existing],
    ):
        search_tool = _make_search_tool(
            document_id="doc-123",
            user_id="user-456",
            k=5,
            client=None,
            accumulated_chunks=accumulated,
        )
        result = await search_tool.ainvoke({"query": "duplicate query"})

    assert "No additional information found" in result


# --- _build_agent tests ---


@pytest.mark.asyncio
async def test_build_agent_returns_agent():
    """_build_agent should return an agent with astream."""
    from src.services.strategies.agentic_rag import _build_agent

    accumulated: list[Document] = []

    with patch("src.services.strategies.agentic_rag.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            llm_model="gpt-4o-mini",
            llm_provider="openai",
            openai_api_key=MagicMock(get_secret_value=lambda: "test-key"),
            llm_provider_base_url="https://api.openai.com/v1",
        )
        with patch(
            "src.services.strategies.agentic_rag.pull_eval_prompt",
            new_callable=AsyncMock,
        ) as mock_prompt:
            mock_sys = MagicMock()
            mock_sys.prompt.template = "You are a research assistant."
            mock_prompt.return_value = MagicMock(messages=[mock_sys])
            agent = await _build_agent(
                query="What is the answer?",
                document_id="doc-123",
                user_id="user-456",
                k=5,
                client=None,
                accumulated_chunks=accumulated,
            )

    assert agent is not None
    assert hasattr(agent, "astream")


# --- execute tests ---


@pytest.mark.asyncio
async def test_execute_yields_chunks_then_tokens_then_citations():
    """Execute should yield ChunksEvent, TokenEvents, then CitationsEvent."""
    from src.services.strategies.agentic_rag import execute

    mock_chunk = Document(page_content="Answer is 42.", metadata={"page": 5})

    async def fake_build_agent(
        *, query, document_id, user_id, k, client, accumulated_chunks
    ):
        async def fake_astream(input_dict, config=None, stream_mode=None):
            accumulated_chunks.append(mock_chunk)
            for token in ["The ", "answer ", "is ", "42. [p. 5]"]:
                yield AIMessageChunk(content=token), {"langgraph_node": "agent"}

        return MagicMock(astream=fake_astream)

    with patch(
        "src.services.strategies.agentic_rag._build_agent",
        side_effect=fake_build_agent,
    ):
        events = []
        async for event in execute("What is the answer?", "doc-123", "user-456"):  # ty: ignore[invalid-argument-type]
            events.append(event)

        assert isinstance(events[0], ChunksEvent)
        assert len(events[0].chunks) == 1
        assert events[0].chunks[0].page_content == "Answer is 42."
        assert events[0].chunks[0].page == 5

        for e in events[1:-1]:
            assert isinstance(e, TokenEvent)

        assert isinstance(events[-1], CitationsEvent)
        assert events[-1].citations[0].page == 5


@pytest.mark.asyncio
async def test_execute_yields_empty_chunks_when_no_results():
    """Execute should yield empty ChunksEvent when agent finds nothing."""
    from src.services.strategies.agentic_rag import execute

    async def fake_build_agent(
        *, query, document_id, user_id, k, client, accumulated_chunks
    ):
        async def fake_astream(input_dict, config=None, stream_mode=None):
            return
            yield  # noqa: F841 — makes this an async generator

        return MagicMock(astream=fake_astream)

    with patch(
        "src.services.strategies.agentic_rag._build_agent",
        side_effect=fake_build_agent,
    ):
        events = []
        async for event in execute("Unknown topic?", "doc-123", "user-456"):  # ty: ignore[invalid-argument-type]
            events.append(event)

        assert isinstance(events[0], ChunksEvent)
        assert events[0].chunks == []
        assert isinstance(events[1], TokenEvent)
        assert isinstance(events[2], CitationsEvent)


@pytest.mark.asyncio
async def test_execute_passes_recursion_limit_to_agent():
    """Execute should pass recursion_limit from settings."""
    from src.services.strategies.agentic_rag import execute

    captured_config: dict[str, Any] = {}

    async def fake_build_agent(
        *, query, document_id, user_id, k, client, accumulated_chunks
    ):
        async def fake_astream(input_dict, config=None, stream_mode=None):
            captured_config["config"] = config
            return
            yield  # noqa: F841 — makes this an async generator

        return MagicMock(astream=fake_astream)

    with (
        patch(
            "src.services.strategies.agentic_rag._build_agent",
            side_effect=fake_build_agent,
        ),
        patch("src.services.strategies.agentic_rag.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(agentic_rag_max_iterations=15)

        async for _ in execute("test?", "doc-123", "user-456"):  # ty: ignore[invalid-argument-type]
            pass

    assert captured_config["config"]["recursion_limit"] == 15


@pytest.mark.asyncio
async def test_execute_skips_tool_call_chunks():
    """Execute should skip AIMessageChunks that contain tool_call_chunks."""
    from src.services.strategies.agentic_rag import execute

    mock_chunk = Document(page_content="Some info.", metadata={"page": 1})

    async def fake_build_agent(
        *, query, document_id, user_id, k, client, accumulated_chunks
    ):
        async def fake_astream(input_dict, config=None, stream_mode=None):
            accumulated_chunks.append(mock_chunk)
            tool_msg = AIMessageChunk(content="")
            tool_msg.tool_call_chunks = [{"name": "search_docs", "args": "{}"}]  # ty:ignore[invalid-assignment]
            yield tool_msg, {"langgraph_node": "agent"}
            yield AIMessageChunk(content="Final answer."), {"langgraph_node": "agent"}

        return MagicMock(astream=fake_astream)

    with patch(
        "src.services.strategies.agentic_rag._build_agent",
        side_effect=fake_build_agent,
    ):
        events = []
        async for event in execute("test?", "doc-123", "user-456"):  # ty: ignore[invalid-argument-type]
            events.append(event)

        token_events = [e for e in events if isinstance(e, TokenEvent)]
        assert len(token_events) == 1
        assert token_events[0].token == "Final answer."
