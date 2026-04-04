"""Tests for the eval target function."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.schemas.chat import Citation, CitationsEvent, TokenEvent


@pytest.mark.asyncio
async def test_target_returns_expected_shape() -> None:
    """Target function returns dict with answer, documents, and citations keys."""
    mock_chunks = [
        Document(page_content="chunk one", metadata={"page": 1}),
        Document(page_content="chunk two", metadata={"page": 3}),
    ]

    fake_events = [
        TokenEvent(token="The answer is "),
        TokenEvent(token="42 [p. 1]."),
        CitationsEvent(citations=[Citation(page=1)]),
    ]

    async def fake_stream(query: str, chunks: list[Document]):
        for event in fake_events:
            yield event

    with (
        patch(
            "src.eval.target.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=mock_chunks,
        ) as mock_retrieve,
        patch(
            "src.eval.target.stream_answer_with_citations",
            side_effect=fake_stream,
        ),
    ):
        from src.eval.target import target

        result = await target(
            {
                "question": "What is the answer?",
                "document_id": "doc-123",
                "user_id": "user-456",
            }
        )

    # Verify output shape
    assert isinstance(result, dict)
    assert "answer" in result
    assert "documents" in result
    assert "citations" in result

    # Verify answer is assembled from token events
    assert result["answer"] == "The answer is 42 [p. 1]."

    # Verify documents are serialised correctly
    assert len(result["documents"]) == 2
    assert result["documents"][0] == {"page_content": "chunk one", "page": 1}
    assert result["documents"][1] == {"page_content": "chunk two", "page": 3}

    # Verify citations are serialised correctly
    assert result["citations"] == [{"page": 1}]

    # Verify retrieve_chunks was called with correct args and no auth client
    mock_retrieve.assert_called_once_with(
        query="What is the answer?",
        document_id="doc-123",
        user_id="user-456",
    )


@pytest.mark.asyncio
async def test_target_with_no_chunks_returns_fallback() -> None:
    """Target returns fallback answer when no chunks are retrieved."""
    fallback_text = "I don't have enough information to answer that question."

    fake_events = [
        TokenEvent(token=fallback_text),
        CitationsEvent(citations=[]),
    ]

    async def fake_stream(query: str, chunks: list[Document]):
        for event in fake_events:
            yield event

    with (
        patch(
            "src.eval.target.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.eval.target.stream_answer_with_citations",
            side_effect=fake_stream,
        ),
    ):
        from src.eval.target import target

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
