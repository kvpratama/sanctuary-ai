"""Unit tests for the dynamic prompt management module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from src.prompts.manager import (
    CORRECTNESS_PROMPT_HARDCODED,
    FALLBACK_PROMPTS,
    pull_eval_prompt,
)


@patch("src.prompts.manager._get_client")
def test_pull_eval_prompt_success(mock_get_client: MagicMock) -> None:
    """Test that pull_eval_prompt successfully retrieves from LangSmith."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_prompt = ChatPromptTemplate.from_template("Hello {name}!")
    mock_client.pull_prompt.return_value = mock_prompt

    result = pull_eval_prompt("sanctuary-test-prompt")  # ty: ignore[invalid-argument-type]

    assert result == mock_prompt
    mock_client.pull_prompt.assert_called_once_with("sanctuary-test-prompt")


@patch("src.prompts.manager._get_client")
def test_pull_eval_prompt_fallback(mock_get_client: MagicMock) -> None:
    """Test that pull_eval_prompt falls back to hardcoded templates on failure."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Simulate a network error or missing prompt
    mock_client.pull_prompt.side_effect = Exception("LangSmith API Error")

    # The known fallback name
    name = "sanctuary-eval-correctness"

    result = pull_eval_prompt(name)  # ty: ignore[invalid-argument-type]

    # Verify fallback was used
    assert result == CORRECTNESS_PROMPT_HARDCODED
    assert result == FALLBACK_PROMPTS[name]
    mock_client.pull_prompt.assert_called_once_with(name)


@patch("src.prompts.manager._get_client")
def test_pull_eval_prompt_fallback_missing(mock_get_client: MagicMock) -> None:
    """Test ValueError is raised when pull fails and no fallback exists."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.pull_prompt.side_effect = Exception("LangSmith API Error")

    with pytest.raises(ValueError, match="Prompt non-existent pull failed"):
        pull_eval_prompt("non-existent")  # ty: ignore[invalid-argument-type]
