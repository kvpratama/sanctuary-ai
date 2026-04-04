"""Tests for the eval correctness evaluator."""

from unittest.mock import MagicMock, patch


def test_correctness_returns_true_when_grader_says_correct() -> None:
    """Correctness evaluator returns True when the LLM grades the answer correct."""
    mock_settings = MagicMock()
    mock_settings.eval_llm_model = "gpt-4o-mini"
    mock_settings.eval_llm_provider = "openai"
    mock_settings.eval_llm_provider_base_url = "https://api.openai.com/v1"
    mock_settings.eval_llm_api_key = None

    mock_grader = MagicMock()
    mock_grader.invoke.return_value = {
        "explanation": "The answer matches.",
        "correct": True,
    }

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_grader

    with (
        patch("src.eval.evaluators.get_settings", return_value=mock_settings),
        patch("src.eval.evaluators.init_chat_model", return_value=mock_llm),
    ):
        from src.eval.evaluators import correctness

        result = correctness(
            inputs={"question": "What is 2+2?"},
            outputs={"answer": "4"},
            reference_outputs={"answer": "4"},
        )

    assert result is True


def test_correctness_returns_false_when_grader_says_incorrect() -> None:
    """Correctness evaluator returns False when the LLM grades the answer wrong."""
    mock_settings = MagicMock()
    mock_settings.eval_llm_model = "gpt-4o-mini"
    mock_settings.eval_llm_provider = "openai"
    mock_settings.eval_llm_provider_base_url = "https://api.openai.com/v1"
    mock_settings.eval_llm_api_key = None

    mock_grader = MagicMock()
    mock_grader.invoke.return_value = {
        "explanation": "The answer is wrong.",
        "correct": False,
    }

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_grader

    with (
        patch("src.eval.evaluators.get_settings", return_value=mock_settings),
        patch("src.eval.evaluators.init_chat_model", return_value=mock_llm),
    ):
        from src.eval.evaluators import correctness

        result = correctness(
            inputs={"question": "What is 2+2?"},
            outputs={"answer": "5"},
            reference_outputs={"answer": "4"},
        )

    assert result is False


def test_correctness_uses_eval_api_key_when_set() -> None:
    """Correctness evaluator passes the eval-specific API key to init_chat_model."""
    mock_settings = MagicMock()
    mock_settings.eval_llm_model = "gpt-4o-mini"
    mock_settings.eval_llm_provider = "openai"
    mock_settings.eval_llm_provider_base_url = "https://api.openai.com/v1"
    mock_settings.eval_llm_api_key = MagicMock()
    mock_settings.eval_llm_api_key.get_secret_value.return_value = "eval-key-123"

    mock_grader = MagicMock()
    mock_grader.invoke.return_value = {"explanation": "ok", "correct": True}

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_grader

    with (
        patch("src.eval.evaluators.get_settings", return_value=mock_settings),
        patch(
            "src.eval.evaluators.init_chat_model", return_value=mock_llm
        ) as mock_init,
    ):
        from src.eval.evaluators import correctness

        correctness(
            inputs={"question": "q"},
            outputs={"answer": "a"},
            reference_outputs={"answer": "a"},
        )

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args[1]
    assert call_kwargs["api_key"] == "eval-key-123"
