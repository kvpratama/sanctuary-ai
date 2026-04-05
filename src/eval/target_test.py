"""Tests for the eval target function."""

from collections.abc import AsyncGenerator
from typing import Union
from unittest.mock import patch

import pytest

from src.eval.target import target
from src.schemas.chat import (
    ChunksEvent,
    Citation,
    CitationsEvent,
    RetrievedChunk,
    TokenEvent,
)

StreamEvent = Union[ChunksEvent, TokenEvent, CitationsEvent]


@pytest.mark.asyncio
async def test_target_returns_expected_shape() -> None:
    """Target function returns dict with answer, documents, and citations."""

    async def fake_pipeline(
        query: str, document_id: str, user_id: str, k: int = 5, *, client=None
    ) -> AsyncGenerator[StreamEvent, None]:
        yield ChunksEvent(
            chunks=[
                RetrievedChunk(page_content="chunk one", page=1),
                RetrievedChunk(page_content="chunk two", page=3),
            ]
        )
        yield TokenEvent(token="The answer is ")
        yield TokenEvent(token="42 [p. 1].")
        yield CitationsEvent(citations=[Citation(page=1)])

    with patch(
        "src.eval.target.stream_rag_pipeline", side_effect=fake_pipeline
    ) as mock_pipeline:
        result = await target(
            {
                "question": "What is the answer?",
                "document_id": "doc-123",
                "user_id": "user-456",
            }
        )

    assert isinstance(result, dict)
    assert result["answer"] == "The answer is 42 [p. 1]."
    assert result["documents"] == [
        RetrievedChunk(page_content="chunk one", page=1),
        RetrievedChunk(page_content="chunk two", page=3),
    ]
    assert result["citations"] == [{"page": 1}]

    mock_pipeline.assert_called_once_with(
        query="What is the answer?",
        document_id="doc-123",
        user_id="user-456",
    )


@pytest.mark.asyncio
async def test_target_with_no_chunks_returns_fallback() -> None:
    """Target returns fallback answer when no chunks are retrieved."""
    fallback_text = "I don't have enough information to answer that question."

    async def fake_pipeline(
        query: str, document_id: str, user_id: str, k: int = 5, *, client=None
    ) -> AsyncGenerator[StreamEvent, None]:
        yield ChunksEvent(chunks=[])
        yield TokenEvent(token=fallback_text)
        yield CitationsEvent(citations=[])

    with patch("src.eval.target.stream_rag_pipeline", side_effect=fake_pipeline):
        result = await target(
            {
                "question": "unknown",
                "document_id": "doc-x",
                "user_id": "user-x",
            }
        )

    assert result["answer"] == fallback_text
    assert result["documents"] == []
    assert result["citations"] == []
