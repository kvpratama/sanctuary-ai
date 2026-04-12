"""Unit tests for the dynamic prompt management module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langsmith.utils import LangSmithAPIError, LangSmithNotFoundError

from src.prompts.manager import (
    CORRECTNESS_PROMPT_HARDCODED,
    FALLBACK_PROMPTS,
    pull_eval_prompt,
    push_eval_prompts,
)


@pytest.mark.asyncio
@patch("src.prompts.manager._get_client")
async def test_pull_eval_prompt_success(mock_get_client: MagicMock) -> None:
    """Test that pull_eval_prompt successfully retrieves from LangSmith."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_prompt = ChatPromptTemplate.from_template("Hello {name}!")
    mock_client.pull_prompt.return_value = mock_prompt

    result = await pull_eval_prompt("sanctuary-test-prompt")

    assert result == mock_prompt
    mock_client.pull_prompt.assert_called_once_with("sanctuary-test-prompt")


@pytest.mark.asyncio
@patch("src.prompts.manager._get_client")
async def test_pull_eval_prompt_fallback(mock_get_client: MagicMock) -> None:
    """Test that pull_eval_prompt falls back to hardcoded templates on failure."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Simulate a network error or missing prompt
    mock_client.pull_prompt.side_effect = LangSmithAPIError("LangSmith API Error")

    # The known fallback name
    name = "sanctuary-eval-correctness"

    result = await pull_eval_prompt(name)

    # Verify fallback was used
    assert result == CORRECTNESS_PROMPT_HARDCODED
    assert result == FALLBACK_PROMPTS[name]
    mock_client.pull_prompt.assert_called_once_with(name)


@pytest.mark.asyncio
@patch("src.prompts.manager._get_client")
async def test_pull_eval_prompt_fallback_missing(mock_get_client: MagicMock) -> None:
    """Test ValueError is raised when pull fails and no fallback exists."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.pull_prompt.side_effect = LangSmithAPIError("LangSmith API Error")

    with pytest.raises(ValueError, match="Prompt non-existent pull failed"):
        await pull_eval_prompt("non-existent")


@patch("src.prompts.manager._get_client")
def test_push_eval_prompts_skips_existing(mock_get_client: MagicMock) -> None:
    """Test that push skips prompts that already exist in LangSmith."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # pull_prompt succeeds → prompt already exists
    mock_client.pull_prompt.return_value = MagicMock()

    push_eval_prompts()

    assert mock_client.pull_prompt.call_count == len(FALLBACK_PROMPTS)
    mock_client.push_prompt.assert_not_called()


@patch("src.prompts.manager._get_client")
def test_push_eval_prompts_pushes_new(mock_get_client: MagicMock) -> None:
    """Test that push creates prompts that don't exist yet."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.pull_prompt.side_effect = LangSmithNotFoundError("not found")
    mock_client.push_prompt.return_value = "https://smith.langchain.com/prompts/test"

    push_eval_prompts()

    assert mock_client.push_prompt.call_count == len(FALLBACK_PROMPTS)
    for name, prompt in FALLBACK_PROMPTS.items():
        mock_client.push_prompt.assert_any_call(name, object=prompt)


def test_query_rewrite_fallback_prompt_exists() -> None:
    """A fallback prompt is registered for sanctuary-query-rewrite."""
    assert "sanctuary-query-rewrite" in FALLBACK_PROMPTS


def test_query_rewrite_prompt_has_query_variable() -> None:
    """The query rewrite prompt template accepts a 'query' input variable."""
    prompt = FALLBACK_PROMPTS["sanctuary-query-rewrite"]
    assert "query" in prompt.input_variables


@patch("src.prompts.manager._get_client")
def test_push_eval_prompts_logs_error_on_failure(
    mock_get_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that push logs an error when an unexpected exception occurs."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.pull_prompt.side_effect = LangSmithAPIError("connection refused")

    push_eval_prompts()

    assert mock_client.push_prompt.call_count == 0
    assert any("Error pushing" in record.message for record in caplog.records)
