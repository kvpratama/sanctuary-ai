"""Jury-of-judges orchestration for multi-LLM evaluation.

Runs each evaluator prompt through multiple LLM judges and applies
Minority Veto aggregation: score is 1 only if ALL judges pass.
"""

import asyncio
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langsmith.evaluation import EvaluationResult
from pydantic import SecretStr

from src.config import JudgeConfig, get_settings
from src.prompts.manager import pull_eval_prompt

_judge_grader_cache: dict[str, Runnable] = {}
_judge_grader_lock: asyncio.Lock = asyncio.Lock()


async def _build_judge_grader(
    judge: JudgeConfig, prompt_name: str, schema: type
) -> Runnable:
    """Return a cached grader chain for a specific judge configuration.

    Builds the chain on first call for each
    ``(model, provider, api_key_field, base_url, prompt_name)``
    combination and caches it. An ``asyncio.Lock`` ensures only one coroutine
    performs initialization.

    Args:
        judge: The judge configuration specifying model, provider, and key.
        prompt_name: LangSmith prompt hub name to pull.
        schema: TypedDict class for structured output.

    Returns:
        A runnable chain that produces dicts matching ``schema``.

    Raises:
        ValueError: If ``api_key_field`` does not exist on Settings or is not
            a ``SecretStr``.
    """
    cache_key = f"{judge.model}:{judge.provider}:{judge.api_key_field}:{judge.base_url}:{prompt_name}"
    if cache_key in _judge_grader_cache:
        return _judge_grader_cache[cache_key]

    async with _judge_grader_lock:
        if cache_key in _judge_grader_cache:
            return _judge_grader_cache[cache_key]

        settings = get_settings()

        api_key_secret = getattr(settings, judge.api_key_field, None)
        if api_key_secret is None:
            raise ValueError(
                f"EVAL_JURY_JUDGES: api_key_field '{judge.api_key_field}' "
                f"does not exist on Settings."
            )
        if not isinstance(api_key_secret, SecretStr):
            raise ValueError(
                f"EVAL_JURY_JUDGES: api_key_field '{judge.api_key_field}' "
                f"is not a SecretStr field (got {type(api_key_secret).__name__}). "
                f"It must reference a SecretStr field on Settings."
            )
        api_key = api_key_secret.get_secret_value()

        llm = init_chat_model(
            model=judge.model,
            model_provider=judge.provider,
            api_key=api_key,
            base_url=judge.base_url or None,
            temperature=0,
        )

        prompt = await pull_eval_prompt(prompt_name)
        _judge_grader_cache[cache_key] = prompt | llm.with_structured_output(schema)
        return _judge_grader_cache[cache_key]


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
        """Run a single judge and return its config with the grade dict.

        Args:
            judge: The judge configuration to evaluate with.

        Returns:
            A tuple of the judge config and its structured grade output.
        """
        grader = await _build_judge_grader(judge, prompt_name, schema)
        return judge, await grader.ainvoke(invoke_kwargs)

    results = await asyncio.gather(
        *[_grade_with_judge(j) for j in judges], return_exceptions=True
    )

    explanations: list[str] = []
    all_pass = True
    for judge, result_item in zip(judges, results):
        if isinstance(result_item, Exception):
            explanations.append(f"[{judge.model}] fail: exception: {result_item}")
            all_pass = False
        else:
            _, grade = result_item
            explanations.append(
                f"[{judge.model}] {'pass' if grade[score_field] else 'fail'}: {grade['explanation']}"
            )
            if not grade[score_field]:
                all_pass = False

    return EvaluationResult(
        key=key,
        score=1 if all_pass else 0,
        comment="- " + "\n- ".join(explanations),
    )
