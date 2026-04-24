"""Unit tests for the step-back RAG strategy."""

from collections.abc import AsyncGenerator
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
from src.services.strategies.step_back import execute, generate_step_back_query


@pytest.mark.asyncio
async def test_generate_step_back_query_returns_llm_response() -> None:
    """generate_step_back_query sends query to LLM and returns the broader question."""
    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    mock_response = MagicMock()
    mock_response.content = "What are the key concepts of machine learning?"

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch(
            "src.services.strategies.step_back.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.services.strategies.step_back.init_chat_model",
            return_value=MagicMock(),
        ),
        patch(
            "src.services.strategies.step_back.pull_eval_prompt",
            new_callable=AsyncMock,
            return_value=mock_prompt,
        ),
    ):
        result = await generate_step_back_query(
            "What is the bias-variance tradeoff in gradient boosting?"  # ty: ignore[invalid-argument-type]
        )

    assert result == "What are the key concepts of machine learning?"
    mock_chain.ainvoke.assert_called_once_with(
        {"query": "What is the bias-variance tradeoff in gradient boosting?"}
    )


@pytest.mark.asyncio
async def test_generate_step_back_query_falls_back_on_empty_response() -> None:
    """generate_step_back_query returns original query if LLM returns empty string."""
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
            "src.services.strategies.step_back.get_settings",
            return_value=mock_settings,
        ),
        patch("src.services.strategies.step_back.init_chat_model"),
        patch(
            "src.services.strategies.step_back.pull_eval_prompt",
            new_callable=AsyncMock,
            return_value=mock_prompt,
        ),
    ):
        result = await generate_step_back_query("original question")  # ty: ignore[invalid-argument-type]

    assert result == "original question"


@pytest.mark.asyncio
async def test_generate_step_back_query_falls_back_on_exception() -> None:
    """generate_step_back_query returns original query when pull_eval_prompt raises."""
    with (
        patch(
            "src.services.strategies.step_back.get_settings",
            return_value=MagicMock(
                llm_model="gpt-4o-mini",
                llm_provider="openai",
                openai_api_key=SecretStr("fake-key"),
                llm_provider_base_url="https://api.openai.com/v1",
            ),
        ),
        patch("src.services.strategies.step_back.init_chat_model"),
        patch(
            "src.services.strategies.step_back.pull_eval_prompt",
            new_callable=AsyncMock,
            side_effect=ValueError("Prompt pull failed and no fallback exists."),
        ),
    ):
        result = await generate_step_back_query("original question")  # ty: ignore[invalid-argument-type]

    assert result == "original question"


@pytest.mark.asyncio
async def test_execute_generates_step_back_retrieves_both_fuses_and_streams() -> None:
    """execute generates step-back query, retrieves for both, fuses with RRF, streams answer."""
    original_chunks = [
        Document(page_content="specific content", metadata={"page": 2}),
    ]
    step_back_chunks = [
        Document(page_content="broad context", metadata={"page": 5}),
    ]
    fused_chunks = [
        Document(page_content="specific content", metadata={"page": 2}),
        Document(page_content="broad context", metadata={"page": 5}),
    ]

    fake_events = [
        TokenEvent(token="Answer."),
        CitationsEvent(citations=[Citation(page=2)]),
    ]

    async def fake_stream(
        query: str, chunks: list[Document]
    ) -> AsyncGenerator[TokenEvent | CitationsEvent, None]:
        for event in fake_events:
            yield event

    async def mock_retrieve(
        query: str,
        document_id: str,
        user_id: str,
        k: int = 5,
        *,
        client: object = None,
    ) -> list[Document]:
        if query == "What is the bias-variance tradeoff in gradient boosting?":
            return original_chunks
        return step_back_chunks

    with (
        patch(
            "src.services.strategies.step_back.generate_step_back_query",
            new_callable=AsyncMock,
            return_value="What are the key concepts of machine learning?",
        ) as mock_gen,
        patch(
            "src.services.strategies.step_back.retrieve_chunks",
            side_effect=mock_retrieve,
        ) as mock_retrieve_fn,
        patch(
            "src.services.strategies.step_back.fuse_rrf",
            return_value=fused_chunks,
        ) as mock_fuse,
        patch(
            "src.services.strategies.step_back.stream_answer_with_citations",
            side_effect=fake_stream,
        ) as mock_stream,
    ):
        results = [
            event
            async for event in execute(
                query="What is the bias-variance tradeoff in gradient boosting?",  # ty: ignore[invalid-argument-type]
                document_id="doc-1",  # ty: ignore[invalid-argument-type]
                user_id="user-1",  # ty: ignore[invalid-argument-type]
                k=5,  # ty: ignore[invalid-argument-type]
            )
        ]

    mock_gen.assert_called_once_with(
        query="What is the bias-variance tradeoff in gradient boosting?"
    )

    assert mock_retrieve_fn.call_count == 2
    mock_retrieve_fn.assert_any_call(
        query="What is the bias-variance tradeoff in gradient boosting?",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
    )
    mock_retrieve_fn.assert_any_call(
        query="What are the key concepts of machine learning?",
        document_id="doc-1",
        user_id="user-1",
        k=5,
        client=None,
    )

    mock_fuse.assert_called_once_with([original_chunks, step_back_chunks], max_chunks=5)

    mock_stream.assert_called_once()
    call_kwargs = mock_stream.call_args
    assert (
        call_kwargs[1]["query"]
        == "What is the bias-variance tradeoff in gradient boosting?"
    )

    assert len(results) == 3
    assert isinstance(results[0], ChunksEvent)
    assert results[0].chunks == [
        RetrievedChunk(page_content="specific content", page=2),
        RetrievedChunk(page_content="broad context", page=5),
    ]
    assert results[1] == TokenEvent(token="Answer.")
    assert isinstance(results[2], CitationsEvent)
