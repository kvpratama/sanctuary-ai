"""LLM-as-judge evaluators for RAG correctness scoring.

Used by ``src.eval.run`` via LangSmith ``evaluate()``.
"""

from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model

from src.config import get_settings


class CorrectnessGrade(TypedDict):
    """Structured output schema for the correctness grader.

    Attributes:
        explanation: Brief reasoning for the grade.
        correct: Whether the actual answer is correct.
    """

    explanation: Annotated[str, "Brief reasoning for the grade"]
    correct: Annotated[bool, "Whether the actual answer is correct"]


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge evaluator that checks answer correctness.

    Compares the target function's answer against the expected answer
    from the dataset using an LLM grader with structured output.

    Args:
        inputs: The dataset example inputs (contains ``question``).
        outputs: The target function's output (contains ``answer``).
        reference_outputs: The expected output (contains ``answer``).

    Returns:
        A dict with ``score`` (1 or 0) and ``comment`` (explanation).
    """
    settings = get_settings()

    api_key = (
        settings.eval_llm_api_key.get_secret_value()
        if settings.eval_llm_api_key
        else None
    )

    llm = init_chat_model(
        model=settings.eval_llm_model,
        model_provider=settings.eval_llm_provider,
        api_key=api_key,
        base_url=settings.eval_llm_provider_base_url,
        temperature=0,
    )

    grader = llm.with_structured_output(CorrectnessGrade)

    grade: CorrectnessGrade = grader.invoke(  # ty: ignore[invalid-assignment]
        f"Grade whether the ACTUAL answer is correct given the EXPECTED answer.\n\n"
        f"Question: {inputs['question']}\n"
        f"Expected: {reference_outputs['answer']}\n"
        f"Actual: {outputs['answer']}\n\n"
        f"Respond with a brief explanation and whether the answer is correct."
    )

    return {"score": 1 if grade["correct"] else 0, "comment": grade["explanation"]}
