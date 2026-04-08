"""Tests for the jury-of-judges evaluation orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langsmith.evaluation import EvaluationResult
from pydantic import SecretStr

from src.config import JudgeConfig
from src.eval.jury import _build_judge_grader, _judge_grader_cache, minority_veto


def _make_judges() -> list[JudgeConfig]:
    """Return a two-judge jury config for testing."""
    return [
        JudgeConfig(model="model-a", provider="openai", api_key_field="openai_api_key"),
        JudgeConfig(model="model-b", provider="openai", api_key_field="openai_api_key"),
    ]


@pytest.mark.asyncio
async def test_minority_veto_all_pass() -> None:
    """minority_veto returns score=1 when every judge passes."""
    mock_grader = MagicMock()
    mock_grader.ainvoke = AsyncMock(
        return_value={"explanation": "Looks good", "correct": True}
    )

    with patch("src.eval.jury._build_judge_grader", return_value=mock_grader):
        result = await minority_veto(
            key="correctness",
            prompt_name="sanctuary-eval-correctness",
            schema=dict,
            invoke_kwargs={"question": "q", "expected": "a", "actual": "a"},
            score_field="correct",
            judges=_make_judges(),
        )

    assert isinstance(result, EvaluationResult)
    assert result.score == 1
    assert result.comment
    assert "model-a" in result.comment
    assert "model-b" in result.comment


@pytest.mark.asyncio
async def test_minority_veto_one_vetoes() -> None:
    """minority_veto returns score=0 when any single judge fails."""
    pass_grade = {"explanation": "Correct", "correct": True}
    fail_grade = {"explanation": "Wrong", "correct": False}

    mock_grader_pass = MagicMock()
    mock_grader_pass.ainvoke = AsyncMock(return_value=pass_grade)
    mock_grader_fail = MagicMock()
    mock_grader_fail.ainvoke = AsyncMock(return_value=fail_grade)

    async def _fake_build(judge, prompt_name, schema):
        """Return a passing grader for model-a and a failing one otherwise."""
        if judge.model == "model-a":
            return mock_grader_pass
        return mock_grader_fail

    with patch("src.eval.jury._build_judge_grader", side_effect=_fake_build):
        result = await minority_veto(
            key="correctness",
            prompt_name="sanctuary-eval-correctness",
            schema=dict,
            invoke_kwargs={"question": "q", "expected": "a", "actual": "b"},
            score_field="correct",
            judges=_make_judges(),
        )

    assert result.score == 0
    assert result.comment
    assert "Wrong" in result.comment


@pytest.mark.asyncio
async def test_minority_veto_all_fail() -> None:
    """minority_veto returns score=0 when all judges fail."""
    mock_grader = MagicMock()
    mock_grader.ainvoke = AsyncMock(
        return_value={"explanation": "Incorrect", "correct": False}
    )

    with patch("src.eval.jury._build_judge_grader", return_value=mock_grader):
        result = await minority_veto(
            key="correctness",
            prompt_name="sanctuary-eval-correctness",
            schema=dict,
            invoke_kwargs={"question": "q", "expected": "a", "actual": "x"},
            score_field="correct",
            judges=_make_judges(),
        )

    assert result.score == 0


@pytest.mark.asyncio
async def test_build_judge_grader_resolves_api_key() -> None:
    """_build_judge_grader reads the API key from the Settings field named in api_key_field."""
    mock_settings = MagicMock()
    mock_settings.openai_api_key = SecretStr("sk-jury-key")

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=MagicMock())

    judge = JudgeConfig(
        model="gpt-4o", provider="openai", api_key_field="openai_api_key"
    )

    _judge_grader_cache.clear()

    with (
        patch("src.eval.jury.get_settings", return_value=mock_settings),
        patch("src.eval.jury.init_chat_model", return_value=mock_llm) as mock_init,
        patch("src.eval.jury.pull_eval_prompt", return_value=mock_prompt),
    ):
        await _build_judge_grader(judge, "sanctuary-eval-correctness", dict)

    mock_init.assert_called_once()
    assert mock_init.call_args[1]["api_key"] == "sk-jury-key"
    assert mock_init.call_args[1]["model"] == "gpt-4o"
    assert mock_init.call_args[1]["model_provider"] == "openai"


@pytest.mark.asyncio
async def test_minority_veto_judge_exception_returns_score_zero() -> None:
    """minority_veto returns score=0 and includes failure message when a judge raises."""
    pass_grade = {"explanation": "Correct", "correct": True}

    mock_grader_pass = MagicMock()
    mock_grader_pass.ainvoke = AsyncMock(return_value=pass_grade)

    mock_grader_fail = MagicMock()
    mock_grader_fail.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    async def _fake_build(judge, prompt_name, schema):
        """Return a passing grader for model-a and a failing one for model-b."""
        if judge.model == "model-a":
            return mock_grader_pass
        return mock_grader_fail

    with patch("src.eval.jury._build_judge_grader", side_effect=_fake_build):
        result = await minority_veto(
            key="correctness",
            prompt_name="sanctuary-eval-correctness",
            schema=dict,
            invoke_kwargs={"question": "q", "expected": "a", "actual": "a"},
            score_field="correct",
            judges=_make_judges(),
        )

    assert result.score == 0
    assert result.comment is not None
    assert "model-b" in result.comment
    assert "exception" in result.comment.lower()


@pytest.mark.asyncio
async def test_minority_veto_build_grader_exception_returns_score_zero() -> None:
    """minority_veto returns score=0 when _build_judge_grader itself raises."""

    async def _fake_build(judge, prompt_name, schema):
        """Raise for model-b, succeed for model-a."""
        if judge.model == "model-b":
            raise ValueError("bad api key field")
        grader = MagicMock()
        grader.ainvoke = AsyncMock(return_value={"explanation": "OK", "correct": True})
        return grader

    with patch("src.eval.jury._build_judge_grader", side_effect=_fake_build):
        result = await minority_veto(
            key="correctness",
            prompt_name="sanctuary-eval-correctness",
            schema=dict,
            invoke_kwargs={"question": "q", "expected": "a", "actual": "a"},
            score_field="correct",
            judges=_make_judges(),
        )

    assert result.score == 0
    assert result.comment is not None
    assert "model-b" in result.comment
    assert "exception" in result.comment.lower()
