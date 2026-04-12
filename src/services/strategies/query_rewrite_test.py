"""Unit tests for the query rewrite RAG strategy."""

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
from src.services.strategies.query_rewrite import execute, rewrite_query


@pytest.mark.asyncio
async def test_rewrite_query_returns_llm_rewritten_text() -> None:
    """rewrite_query sends query to LLM and returns the rewritten text."""
    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "data distribution shift detection monitoring production"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch(
            "src.services.strategies.query_rewrite.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.services.strategies.query_rewrite.init_chat_model",
            return_value=mock_llm,
        ),
        patch(
            "src.services.strategies.query_rewrite.pull_eval_prompt",
            new_callable=AsyncMock,
            return_value=mock_prompt,
        ),
    ):
        result = await rewrite_query(
            "How do I handle data distribution shifts in a live environment?"  # ty: ignore[invalid-argument-type]
        )

    assert result == "data distribution shift detection monitoring production"
    mock_chain.ainvoke.assert_called_once_with(
        {"query": "How do I handle data distribution shifts in a live environment?"}
    )


@pytest.mark.asyncio
async def test_rewrite_query_falls_back_to_original_on_empty_response() -> None:
    """rewrite_query returns original query if LLM returns empty string."""
    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    mock_response = MagicMock()
    mock_response.content = ""

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch(
            "src.services.strategies.query_rewrite.get_settings",
            return_value=mock_settings,
        ),
        patch("src.services.strategies.query_rewrite.init_chat_model"),
        patch(
            "src.services.strategies.query_rewrite.pull_eval_prompt",
            new_callable=AsyncMock,
            return_value=mock_prompt,
        ),
    ):
        result = await rewrite_query("original question")  # ty: ignore[invalid-argument-type]

    assert result == "original question"


@pytest.mark.asyncio
async def test_execute_rewrites_query_then_retrieves_and_streams() -> None:
    """execute rewrites the query, retrieves with rewritten query, streams answer with original query."""
    mock_chunks = [
        Document(page_content="relevant content", metadata={"page": 3}),
    ]

    fake_events = [
        TokenEvent(token="Answer."),
        CitationsEvent(citations=[Citation(page=3)]),
    ]

    async def fake_stream(query: str, chunks: list[Document]) -> None:  # type: ignore[override]  # ty:ignore[invalid-return-type]
        for event in fake_events:
            yield event

    with (
        patch(
            "src.services.strategies.query_rewrite.rewrite_query",
            new_callable=AsyncMock,
            return_value="optimized search query",
        ) as mock_rewrite,
        patch(
            "src.services.strategies.query_rewrite.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=mock_chunks,
        ) as mock_retrieve,
        patch(
            "src.services.strategies.query_rewrite.stream_answer_with_citations",
            side_effect=fake_stream,
        ) as mock_stream,
    ):
        results = [
            event
            async for event in execute(
                query="How do I handle shifts?",  # ty: ignore[invalid-argument-type]
                document_id="doc-1",  # ty: ignore[invalid-argument-type]
                user_id="user-1",  # ty: ignore[invalid-argument-type]
                k=5,  # ty: ignore[invalid-argument-type]
            )
        ]

    mock_rewrite.assert_called_once_with(query="How do I handle shifts?")

    mock_retrieve.assert_called_once_with(
        query="optimized search query",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
    )

    mock_stream.assert_called_once()
    call_kwargs = mock_stream.call_args
    assert call_kwargs[1]["query"] == "How do I handle shifts?"

    assert len(results) == 3
    assert isinstance(results[0], ChunksEvent)
    assert results[0].chunks == [
        RetrievedChunk(page_content="relevant content", page=3)
    ]
    assert results[1] == TokenEvent(token="Answer.")
    assert isinstance(results[2], CitationsEvent)
