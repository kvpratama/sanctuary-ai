"""LLM-as-judge evaluators for RAG correctness scoring.

Used by ``src.eval.run`` via LangSmith ``evaluate()``.
"""

import asyncio
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run

from src.config import get_settings
from src.prompts.manager import pull_eval_prompt

_grader_cache: Runnable | None = None
_grader_lock: asyncio.Lock = asyncio.Lock()


class CorrectnessGrade(TypedDict):
    """Structured output schema for the correctness grader.

    Attributes:
        explanation: Brief reasoning for the grade.
        correct: Whether the actual answer is correct.
    """

    explanation: Annotated[str, "Brief reasoning for the grade"]
    correct: Annotated[bool, "Whether the actual answer is correct"]


async def _get_grader() -> Runnable:
    """Return the LLM grader chain with structured output.

    The chain is built once and cached at module level. An ``asyncio.Lock``
    ensures only one coroutine performs the initialization under concurrency.

    Returns:
        A runnable chain that produces ``CorrectnessGrade`` dicts.
    """
    global _grader_cache  # noqa: PLW0603

    if _grader_cache is not None:
        return _grader_cache

    async with _grader_lock:
        if _grader_cache is not None:
            return _grader_cache

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

        prompt = await pull_eval_prompt("sanctuary-eval-correctness")
        _grader_cache = prompt | llm.with_structured_output(CorrectnessGrade)
        return _grader_cache


def _get_outputs(obj, attr: str) -> dict:
    """Safely extract a dict attribute from either a RunTree or a plain dict."""
    return (
        (getattr(obj, attr, None) or {})
        if hasattr(obj, attr)
        else (obj.get(attr) or {})
    )


async def correctness(run: Run, example: Example | None) -> EvaluationResult:
    """LLM-as-judge evaluator that checks answer correctness.

    Compares the target function's answer against the expected answer
    from the dataset using an LLM grader with structured output.

    Args:
        run: The LangSmith run object containing the target function's outputs
            (must include an ``answer`` key).
        example: The LangSmith dataset example containing inputs and expected
            outputs (inputs must include ``question``, outputs must include
            ``answer``).

    Returns:
        An ``EvaluationResult`` with ``key`` set to ``correctness``, ``score``
        of 1 or 0, and a ``comment`` containing the grader's explanation.
    """

    run_outputs = _get_outputs(run, "outputs")
    example_inputs = _get_outputs(example, "inputs")
    example_outputs = _get_outputs(example, "outputs")

    missing = []
    if "question" not in example_inputs:
        missing.append("question (example.inputs)")
    if "answer" not in example_outputs:
        missing.append("answer (example.outputs)")
    if "answer" not in run_outputs:
        missing.append("answer (run.outputs)")

    if missing:
        return EvaluationResult(
            key="correctness",
            score=0,
            comment=f"Missing required keys: {', '.join(missing)}",
        )

    grader = await _get_grader()

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
