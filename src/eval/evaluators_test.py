"""Tests for the eval evaluators: correctness, relevance, groundedness, and retrieval_relevance."""

from typing import Mapping
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langsmith.schemas import Example, Run

from src.eval.evaluators import (
    correctness,
    groundedness,
    relevance,
    retrieval_relevance,
)
from src.schemas.chat import RetrievedChunk


def _make_run(outputs: Mapping[str, object]) -> Run:
    """Create a minimal mock Run with the given outputs."""
    run = MagicMock(spec=Run)
    run.outputs = outputs
    return run


def _make_example(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> Example:
    """Create a minimal mock Example with the given inputs/outputs."""
    example = MagicMock(spec=Example)
    example.inputs = inputs
    example.outputs = outputs
    return example


@pytest.mark.asyncio
async def test_correctness_returns_true_when_grader_says_correct() -> None:
    """Correctness evaluator returns score 1 when the LLM grades the answer correct."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value={
            "explanation": "The answer matches.",
            "correct": True,
        }
    )

    with patch("src.eval.evaluators._get_grader", return_value=mock_chain):
        result = await correctness(
            run=_make_run({"answer": "4"}),
            example=_make_example({"question": "What is 2+2?"}, {"answer": "4"}),
        )

    assert result.score == 1
    assert result.comment == "The answer matches."


@pytest.mark.asyncio
async def test_correctness_returns_false_when_grader_says_incorrect() -> None:
    """Correctness evaluator returns score 0 when the LLM grades the answer wrong."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value={
            "explanation": "The answer is wrong.",
            "correct": False,
        }
    )

    with patch("src.eval.evaluators._get_grader", return_value=mock_chain):
        result = await correctness(
            run=_make_run({"answer": "5"}),
            example=_make_example({"question": "What is 2+2?"}, {"answer": "4"}),
        )

    assert result.score == 0
    assert result.comment == "The answer is wrong."


@pytest.mark.asyncio
async def test_correctness_uses_eval_api_key_when_set() -> None:
    """Correctness evaluator passes the eval-specific API key to init_chat_model."""
    mock_settings = MagicMock()
    mock_settings.eval_llm_model = "gpt-4o-mini"
    mock_settings.eval_llm_provider = "openai"
    mock_settings.eval_llm_provider_base_url = "https://api.openai.com/v1"
    mock_settings.eval_llm_api_key = MagicMock()
    mock_settings.eval_llm_api_key.get_secret_value.return_value = "eval-key-123"

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()

    # Clear caches so prior calls don't leak into this test.
    from src.config import get_settings
    from src.eval import evaluators

    get_settings.cache_clear()
    evaluators._grader_cache.clear()

    mock_pull = AsyncMock(return_value=MagicMock())

    with (
        patch("src.eval.evaluators.get_settings", return_value=mock_settings),
        patch(
            "src.eval.evaluators.init_chat_model", return_value=mock_llm
        ) as mock_init,
        patch(
            "src.eval.evaluators.pull_eval_prompt",
            mock_pull,
        ),
    ):
        from src.eval.evaluators import CorrectnessGrade, _get_grader

        await _get_grader("correctness", "sanctuary-eval-correctness", CorrectnessGrade)

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args[1]
    assert call_kwargs["api_key"] == "eval-key-123"
    mock_pull.assert_awaited_once_with("sanctuary-eval-correctness")


@pytest.mark.asyncio
async def test_relevance_returns_true_when_grader_says_relevant() -> None:
    """Relevance evaluator returns score 1 when the LLM says the answer is relevant."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value={
            "explanation": "The answer addresses the question.",
            "relevant": True,
        }
    )

    with patch("src.eval.evaluators._get_grader", return_value=mock_chain):
        result = await relevance(
            run=_make_run({"answer": "Python is a programming language."}),
            example=_make_example({"question": "What is Python?"}, {}),
        )

    assert result.key == "relevance"
    assert result.score == 1
    assert result.comment == "The answer addresses the question."


@pytest.mark.asyncio
async def test_relevance_returns_false_when_grader_says_irrelevant() -> None:
    """Relevance evaluator returns score 0 when the LLM says the answer is irrelevant."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value={
            "explanation": "The answer is off-topic.",
            "relevant": False,
        }
    )

    with patch("src.eval.evaluators._get_grader", return_value=mock_chain):
        result = await relevance(
            run=_make_run({"answer": "The sky is blue."}),
            example=_make_example({"question": "What is Python?"}, {}),
        )

    assert result.key == "relevance"
    assert result.score == 0
    assert result.comment == "The answer is off-topic."


@pytest.mark.asyncio
async def test_relevance_missing_keys() -> None:
    """Relevance evaluator returns score 0 when required keys are missing."""
    result = await relevance(
        run=_make_run({}),
        example=_make_example({}, {}),
    )

    assert result.key == "relevance"
    assert result.score == 0
    assert result.comment is not None
    assert "Missing required keys" in result.comment


@pytest.mark.asyncio
async def test_groundedness_returns_true_when_grader_says_grounded() -> None:
    """Groundedness evaluator returns score 1 when answer is grounded in docs."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value={
            "explanation": "All claims are supported by the documents.",
            "grounded": True,
        }
    )

    with patch("src.eval.evaluators._get_grader", return_value=mock_chain):
        result = await groundedness(
            run=_make_run(
                {
                    "answer": "The sky is blue.",
                    "documents": ["The sky appears blue due to Rayleigh scattering."],
                }
            ),
            example=_make_example({}, {}),
        )

    assert result.key == "groundedness"
    assert result.score == 1
    assert result.comment == "All claims are supported by the documents."


@pytest.mark.asyncio
async def test_groundedness_returns_false_when_grader_says_not_grounded() -> None:
    """Groundedness evaluator returns score 0 when answer is not grounded."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value={
            "explanation": "The answer contains claims not in the documents.",
            "grounded": False,
        }
    )

    with patch("src.eval.evaluators._get_grader", return_value=mock_chain):
        result = await groundedness(
            run=_make_run(
                {
                    "answer": "Mars is red.",
                    "documents": ["The sky is blue."],
                }
            ),
            example=_make_example({}, {}),
        )

    assert result.key == "groundedness"
    assert result.score == 0
    assert result.comment == "The answer contains claims not in the documents."


@pytest.mark.asyncio
async def test_groundedness_missing_keys() -> None:
    """Groundedness evaluator returns score 0 when required keys are missing."""
    result = await groundedness(
        run=_make_run({}),
        example=_make_example({}, {}),
    )

    assert result.key == "groundedness"
    assert result.score == 0
    assert result.comment is not None
    assert "Missing required keys" in result.comment


@pytest.mark.asyncio
async def test_retrieval_relevance_returns_true_when_docs_relevant() -> None:
    """Retrieval relevance evaluator returns score 1 when docs are relevant."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value={
            "explanation": "The documents contain relevant information.",
            "relevant": True,
        }
    )

    with patch("src.eval.evaluators._get_grader", return_value=mock_chain):
        result = await retrieval_relevance(
            run=_make_run({"documents": ["Python is a programming language."]}),
            example=_make_example({"question": "What is Python?"}, {}),
        )

    assert result.key == "retrieval_relevance"
    assert result.score == 1
    assert result.comment == "The documents contain relevant information."


@pytest.mark.asyncio
async def test_retrieval_relevance_returns_false_when_docs_irrelevant() -> None:
    """Retrieval relevance evaluator returns score 0 when docs are irrelevant."""
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value={
            "explanation": "The documents are not related to the question.",
            "relevant": False,
        }
    )

    with patch("src.eval.evaluators._get_grader", return_value=mock_chain):
        result = await retrieval_relevance(
            run=_make_run({"documents": ["Recipe for chocolate cake."]}),
            example=_make_example({"question": "What is Python?"}, {}),
        )

    assert result.key == "retrieval_relevance"
    assert result.score == 0
    assert result.comment == "The documents are not related to the question."


@pytest.mark.asyncio
async def test_retrieval_relevance_missing_keys() -> None:
    """Retrieval relevance evaluator returns score 0 when required keys are missing."""
    result = await retrieval_relevance(
        run=_make_run({}),
        example=_make_example({}, {}),
    )

    assert result.key == "retrieval_relevance"
    assert result.score == 0
    assert result.comment is not None
    assert "Missing required keys" in result.comment


@pytest.mark.asyncio
async def test_get_outputs_handles_all_types() -> None:
    """_get_outputs safely extracts attributes from Runs, Examples, dicts, or None."""
    from src.eval.evaluators import _get_outputs

    # Case 1: None
    assert _get_outputs(None, "outputs") == {}

    # Case 2: Dictionary
    data = {"outputs": {"key": "value"}}
    assert _get_outputs(data, "outputs") == {"key": "value"}
    assert _get_outputs(data, "missing") == {}

    # Case 3: Mock Run (getattr)
    run = MagicMock(spec=Run)
    run.outputs = {"a": 1}
    assert _get_outputs(run, "outputs") == {"a": 1}
    assert _get_outputs(run, "missing") == {}

    # Case 4: Mock Example (getattr)
    example = MagicMock(spec=Example)
    example.inputs = {"q": "test"}
    assert _get_outputs(example, "inputs") == {"q": "test"}


def test_format_docs_includes_page_numbers() -> None:
    """_format_docs includes page numbers when RetrievedChunk objects are provided."""
    from src.eval.evaluators import _format_docs

    docs = [
        RetrievedChunk(page_content="Content 1", page=10),
        RetrievedChunk(page_content="Content 2", page=20),
    ]

    formatted = _format_docs(docs)

    assert "Document 1 (Page 10):\nContent 1" in formatted
    assert "Document 2 (Page 20):\nContent 2" in formatted
