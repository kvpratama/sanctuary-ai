"""Jury-of-judges orchestration for multi-LLM evaluation.

Runs each evaluator prompt through multiple LLM judges and applies
Minority Veto aggregation: score is 1 only if ALL judges pass.
"""

import asyncio
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langsmith.evaluation import EvaluationResult

from src.config import JudgeConfig, get_settings
from src.prompts.manager import pull_eval_prompt


async def _build_judge_grader(
    judge: JudgeConfig, prompt_name: str, schema: type
) -> Runnable:
    """Build a grader chain for a specific judge configuration.

    Args:
        judge: The judge configuration specifying model, provider, and key.
        prompt_name: LangSmith prompt hub name to pull.
        schema: TypedDict class for structured output.

    Returns:
        A runnable chain that produces dicts matching ``schema``.
    """
    settings = get_settings()

    api_key_secret = getattr(settings, judge.api_key_field, None)
    api_key = api_key_secret.get_secret_value() if api_key_secret else None

    llm = init_chat_model(
        model=judge.model,
        model_provider=judge.provider,
        api_key=api_key,
        base_url=judge.base_url or None,
        temperature=0,
    )

    prompt = await pull_eval_prompt(prompt_name)
    return prompt | llm.with_structured_output(schema)


async def minority_veto(
    key: str,
    prompt_name: str,
    schema: type,
    invoke_kwargs: dict[str, Any],
    score_field: str,
    judges: list[JudgeConfig],
) -> EvaluationResult:
    """Run all jury judges and aggregate with Minority Veto.

    Score is 1 only if every judge passes. If any single judge
    returns a failing grade, the overall score is 0.

    Args:
        key: Evaluator key name (e.g. ``"correctness"``).
        prompt_name: LangSmith prompt hub name for the grader prompt.
        schema: TypedDict class for structured output.
        invoke_kwargs: Arguments to pass to each grader chain.
        score_field: Key in the grader output dict that holds the boolean grade.
        judges: List of judge configurations to evaluate with.

    Returns:
        An ``EvaluationResult`` with aggregated score and all judges' explanations.
    """

    async def _grade_with_judge(
        judge: JudgeConfig,
    ) -> tuple[JudgeConfig, dict[str, Any]]:
        grader = await _build_judge_grader(judge, prompt_name, schema)
        return judge, await grader.ainvoke(invoke_kwargs)

    results = await asyncio.gather(*[_grade_with_judge(j) for j in judges])

    explanations: list[str] = []
    all_pass = True
    for judge, grade in results:
        explanations.append(
            f"[{judge.model}] {'pass' if grade[score_field] else 'fail'}: {grade['explanation']}"
        )
        if not grade[score_field]:
            all_pass = False

    return EvaluationResult(
        key=key,
        score=1 if all_pass else 0,
        comment=" | ".join(explanations),
    )
