from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.schemas.chat import (
    ChunksEvent,
    Citation,
    CitationsEvent,
    RetrievedChunk,
    StreamEvent,
    TokenEvent,
)
from src.services.retrieval import stream_rag_pipeline


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
            "src.services.strategies.naive_rag.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=mock_chunks,
        ) as mock_retrieve,
        patch(
            "src.services.strategies.naive_rag.stream_answer_with_citations",
            side_effect=fake_stream,
        ),
        patch("src.services.retrieval.get_settings") as mock_settings,
    ):
        mock_settings.return_value.rag_strategy = "naive_rag"
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


@pytest.mark.asyncio
async def test_stream_rag_pipeline_dispatches_to_configured_strategy() -> None:
    """stream_rag_pipeline delegates to the strategy from settings.rag_strategy."""
    fake_events: list[StreamEvent] = [
        ChunksEvent(chunks=[]),
        TokenEvent(token="dispatched"),
        CitationsEvent(citations=[]),
    ]

    async def fake_execute(
        query: str,
        document_id: str,
        user_id: str,
        k: int = 5,
        *,
        client: object | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        for event in fake_events:
            yield event

    mock_settings = MagicMock()
    mock_settings.rag_strategy = "naive_rag"

    with (
        patch("src.services.retrieval.get_settings", return_value=mock_settings),
        patch(
            "src.services.retrieval.get_strategy",
            return_value=fake_execute,
        ) as mock_get_strategy,
    ):
        results = [
            event
            async for event in stream_rag_pipeline(
                query="test",  # ty: ignore[invalid-argument-type]
                document_id="doc-1",  # ty: ignore[invalid-argument-type]
                user_id="user-1",  # ty: ignore[invalid-argument-type]
            )
        ]

    mock_get_strategy.assert_called_once_with("naive_rag")
    assert len(results) == 3
    assert results[1] == TokenEvent(token="dispatched")
