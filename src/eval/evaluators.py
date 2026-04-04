"""LLM-as-judge evaluators for RAG correctness scoring.

Used by ``src.eval.run`` via LangSmith ``evaluate()``.
"""

from functools import lru_cache
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run

from src.config import get_settings


class CorrectnessGrade(TypedDict):
    """Structured output schema for the correctness grader.

    Attributes:
        explanation: Brief reasoning for the grade.
        correct: Whether the actual answer is correct.
    """

    explanation: Annotated[str, "Brief reasoning for the grade"]
    correct: Annotated[bool, "Whether the actual answer is correct"]


GRADING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grading assistant. Grade whether the ACTUAL answer "
            "is correct given the EXPECTED answer. Respond with a brief "
            "explanation and whether the answer is correct.",
        ),
        (
            "human",
            "Question: {question}\nExpected: {expected}\nActual: {actual}",
        ),
    ]
)


@lru_cache(maxsize=1)
def _get_grader() -> Runnable:
    """Return the cached LLM grader chain with structured output.

    The grader is created once and reused across evaluator calls.

    Returns:
        A runnable chain that produces ``CorrectnessGrade`` dicts.
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

    return GRADING_PROMPT | llm.with_structured_output(CorrectnessGrade)


async def correctness(run: Run, example: Example | None) -> EvaluationResult:
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
    if not run.outputs or not example or not example.outputs or not example.inputs:
        return EvaluationResult(
            key="correctness",
            score=0,
            comment="No outputs found",
        )

    run_outputs = (
        run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}
    )
    example_inputs = (
        example.inputs
        if hasattr(example, "inputs")
        else example.get("inputs", {}) or {}
    )
    example_outputs = (
        example.outputs
        if hasattr(example, "outputs")
        else example.get("outputs", {}) or {}
    )

    grader = _get_grader()

    grade: CorrectnessGrade = await grader.ainvoke(
        {
            "question": example_inputs["question"],
            "expected": example_outputs["answer"],
            "actual": run_outputs["answer"],
        }
    )

    return EvaluationResult(
        key="correctness",
        score=1 if grade["correct"] else 0,
        comment=grade["explanation"],
    )
