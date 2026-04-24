"""Unit tests for the multi-query RAG strategy."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr

from src.schemas.chat import (
    ChunksEvent,
    Citation,
    CitationsEvent,
    RetrievedChunk,
    StreamEvent,
    TokenEvent,
)
from src.services.strategies.multi_query import (
    QueryVariants,
    execute,
    generate_query_variants,
)


@pytest.mark.asyncio
async def test_generate_query_variants_returns_parsed_queries() -> None:
    """generate_query_variants returns parsed queries via structured output."""
    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    mock_response = QueryVariants(
        variants=[
            "How to detect data distribution shifts in production ML systems",
            "Methods for handling covariate shift in deployed models",
            "Monitoring and adapting to data drift in live ML pipelines",
        ]
    )

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)

    mock_structured_llm = MagicMock()
    mock_structured_llm.__or__ = MagicMock()

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch(
            "src.services.strategies.multi_query.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.services.strategies.multi_query.init_chat_model",
            return_value=mock_llm,
        ),
        patch(
            "src.services.strategies.multi_query.pull_eval_prompt",
            new_callable=AsyncMock,
            return_value=mock_prompt,
        ),
    ):
        result = await generate_query_variants(
            "How do I handle data distribution shifts?",  # ty: ignore[invalid-argument-type]
            n=3,  # ty: ignore[invalid-argument-type]
        )

    assert len(result) == 3
    assert (
        result[0] == "How to detect data distribution shifts in production ML systems"
    )
    assert result[1] == "Methods for handling covariate shift in deployed models"
    assert result[2] == "Monitoring and adapting to data drift in live ML pipelines"


@pytest.mark.asyncio
async def test_generate_query_variants_falls_back_to_original() -> None:
    """generate_query_variants returns [original] if LLM returns empty variants."""
    mock_settings = MagicMock()
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = SecretStr("fake-key")
    mock_settings.llm_provider_base_url = "https://api.openai.com/v1"

    mock_response = QueryVariants(variants=[])

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)

    mock_structured_llm = MagicMock()
    mock_structured_llm.__or__ = MagicMock()

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch(
            "src.services.strategies.multi_query.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.services.strategies.multi_query.init_chat_model",
            return_value=mock_llm,
        ),
        patch(
            "src.services.strategies.multi_query.pull_eval_prompt",
            new_callable=AsyncMock,
            return_value=mock_prompt,
        ),
    ):
        result = await generate_query_variants("original question", n=3)  # ty: ignore[invalid-argument-type]

    assert result == ["original question"]


@pytest.mark.asyncio
async def test_generate_query_variants_falls_back_on_exception() -> None:
    """generate_query_variants returns [original] when pull_eval_prompt raises."""
    with (
        patch(
            "src.services.strategies.multi_query.get_settings",
            return_value=MagicMock(
                llm_model="gpt-4o-mini",
                llm_provider="openai",
                openai_api_key=SecretStr("fake-key"),
                llm_provider_base_url="https://api.openai.com/v1",
            ),
        ),
        patch("src.services.strategies.multi_query.init_chat_model"),
        patch(
            "src.services.strategies.multi_query.pull_eval_prompt",
            new_callable=AsyncMock,
            side_effect=ValueError("Prompt pull failed and no fallback exists."),
        ),
    ):
        result = await generate_query_variants("original question", n=3)  # ty: ignore[invalid-argument-type]

    assert result == ["original question"]


@pytest.mark.asyncio
async def test_execute_generates_variants_retrieves_fuses_and_streams() -> None:
    """execute generates variants, retrieves for each, fuses with RRF, and streams."""
    chunks_variant_1 = [
        Document(page_content="content A", metadata={"page": 1}),
        Document(page_content="content B", metadata={"page": 2}),
    ]
    chunks_variant_2 = [
        Document(page_content="content B", metadata={"page": 2}),
        Document(page_content="content C", metadata={"page": 3}),
    ]

    call_count = 0

    async def mock_retrieve(
        query: str,
        document_id: str,
        user_id: str,
        k: int = 5,
        *,
        client: Any = None,
    ) -> list[Document]:
        """Return different chunk lists on successive calls."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return chunks_variant_1
        return chunks_variant_2

    fake_events = [
        TokenEvent(token="Merged answer."),
        CitationsEvent(citations=[Citation(page=1)]),
    ]

    async def fake_stream(
        query: str, chunks: list[Document]
    ) -> AsyncIterator[StreamEvent]:
        """Yield pre-built fake events for testing."""
        for event in fake_events:
            yield event

    with (
        patch(
            "src.services.strategies.multi_query.generate_query_variants",
            new_callable=AsyncMock,
            return_value=["variant 1", "variant 2"],
        ) as mock_gen,
        patch(
            "src.services.strategies.multi_query.retrieve_chunks",
            side_effect=mock_retrieve,
        ) as mock_retrieve_fn,
        patch(
            "src.services.strategies.multi_query.stream_answer_with_citations",
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

    # Verify variants were generated from original query
    mock_gen.assert_called_once_with("original question", n=3)

    # Verify retrieve was called for each variant
    assert mock_retrieve_fn.call_count == 2

    # Verify generation used the ORIGINAL query
    mock_stream.assert_called_once()
    call_kwargs = mock_stream.call_args
    assert call_kwargs[1]["query"] == "original question"
    # Verify fused chunks: 3 unique, B ranked first (appears in both lists)
    passed_chunks = call_kwargs[1]["chunks"]
    assert len(passed_chunks) == 3
    assert passed_chunks[0].page_content == "content B"

    # Verify event sequence
    assert len(results) == 3
    assert isinstance(results[0], ChunksEvent)
    assert len(results[0].chunks) == 3
    assert results[0].chunks[0].page_content == "content B"
    assert results[1] == TokenEvent(token="Merged answer.")
    assert isinstance(results[2], CitationsEvent)
