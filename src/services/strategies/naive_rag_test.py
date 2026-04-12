from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from src.schemas.chat import (
    ChunksEvent,
    Citation,
    CitationsEvent,
    RetrievedChunk,
    TokenEvent,
)
from src.services.strategies.naive_rag import execute


@pytest.mark.asyncio
async def test_naive_rag_execute_yields_chunks_then_stream_events():
    """naive_rag.execute yields ChunksEvent then delegates to stream_answer_with_citations."""
    mock_chunks = [
        Document(page_content="chunk one", metadata={"page": 1}),
    ]

    fake_events = [
        TokenEvent(token="Hello "),
        TokenEvent(token="world."),
        CitationsEvent(citations=[Citation(page=1)]),
    ]

    async def fake_stream(query, chunks):
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
    ):
        mock_client = AsyncMock()
        results = [
            event
            async for event in execute(
                query="Hi",  # ty: ignore[invalid-argument-type]
                document_id="doc-1",  # ty: ignore[invalid-argument-type]
                user_id="user-1",  # ty: ignore[invalid-argument-type]
                k=3,  # ty: ignore[invalid-argument-type]
                client=mock_client,
            )
        ]

    assert len(results) == 4

    assert isinstance(results[0], ChunksEvent)
    assert results[0].chunks == [RetrievedChunk(page_content="chunk one", page=1)]

    assert results[1] == TokenEvent(token="Hello ")
    assert results[2] == TokenEvent(token="world.")
    assert isinstance(results[3], CitationsEvent)

    mock_retrieve.assert_called_once_with(
        query="Hi",
        document_id="doc-1",
        user_id="user-1",
        k=3,
        client=mock_client,
    )
