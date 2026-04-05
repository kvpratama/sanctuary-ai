"""Tests for the eval correctness evaluator."""

from typing import Mapping
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langsmith.schemas import Example, Run

from src.eval.evaluators import correctness


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

    # Clear the global get_settings cache so prior calls don't leak into this test.
    from src.config import get_settings

    get_settings.cache_clear()

    with (
        patch("src.eval.evaluators.get_settings", return_value=mock_settings),
        patch(
            "src.eval.evaluators.init_chat_model", return_value=mock_llm
        ) as mock_init,
    ):
        from src.eval.evaluators import _get_grader

        _get_grader.cache_clear()
        _get_grader()

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args[1]
    assert call_kwargs["api_key"] == "eval-key-123"
