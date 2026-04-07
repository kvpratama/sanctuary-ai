"""LLM-as-judge evaluators for RAG correctness scoring.

Used by ``src.eval.run`` via LangSmith ``evaluate()``.
"""

import asyncio
from typing import Annotated, Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run

from src.config import get_settings
from src.eval.jury import minority_veto
from src.prompts.manager import pull_eval_prompt
from src.schemas.chat import RetrievedChunk

_grader_cache: dict[str, Runnable] = {}
_grader_lock: asyncio.Lock = asyncio.Lock()


async def _get_grader(name: str, prompt_name: str, schema: type) -> Runnable:
    """Return a cached LLM grader chain for the given evaluator.

    Builds the chain on first call for each ``name`` and caches it.
    An ``asyncio.Lock`` ensures only one coroutine performs initialization.

    Args:
        name: Unique cache key for this grader (e.g. ``"correctness"``).
        prompt_name: LangSmith prompt hub name to pull.
        schema: TypedDict class for structured output.

    Returns:
        A runnable chain that produces dicts matching ``schema``.
    """
    if name in _grader_cache:
        return _grader_cache[name]

    async with _grader_lock:
        if name in _grader_cache:
            return _grader_cache[name]

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

        prompt = await pull_eval_prompt(prompt_name)
        _grader_cache[name] = prompt | llm.with_structured_output(schema)
        return _grader_cache[name]


def _get_outputs(
    obj: Run | Example | dict[str, Any] | None, attr: str
) -> dict[str, Any]:
    """Safely extract a dict attribute from a Run, Example, or plain dict.

    Args:
        obj: The object to extract from. Accepts a LangSmith Run, Example,
            a plain dict, or None.
        attr: The attribute or key name to look up (e.g. ``"outputs"``).

    Returns:
        The value at ``attr`` as a dict, or an empty dict if the attribute
        is missing, None, or the object itself is None.
    """
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj.get(attr) or {}

    return getattr(obj, attr, None) or {}


def _format_docs(docs: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered text block.

    Args:
        docs: A list of retrieved chunks to format. Each item can be a
            LangChain Document object, a RetrievedChunk, a dict with a
            ``page_content`` key, or any object coercible to string.
            If a plain string is passed, it is returned as-is.

    Returns:
        A single string where each document is separated by a blank line,
        prefixed with a numbered header and an optional page number, e.g.::

            Document 1 (Page 3):
            Some content here

            Document 2:
            More content here
    """
    if isinstance(docs, str):
        return docs

    parts = []
    for i, d in enumerate(docs, 1):
        if hasattr(d, "page_content"):
            content = d.page_content
            if hasattr(d, "page"):
                page = d.page
            elif hasattr(d, "metadata") and isinstance(d.metadata, dict):
                page = d.metadata.get("page")
            else:
                page = None
        elif isinstance(d, dict):
            content = d.get("page_content", str(d))
            page = d.get("page")
        else:
            content = str(d)
            page = None

        page_str = f" (Page {page})" if page is not None else ""
        parts.append(f"Document {i}{page_str}:\n{content}")

    return "\n\n".join(parts)


class CorrectnessGrade(TypedDict):
    """Structured output schema for the correctness grader.

    Attributes:
        explanation: Brief reasoning for the grade.
        correct: Whether the actual answer is correct.
    """

    explanation: Annotated[str, "Brief reasoning for the grade"]
    correct: Annotated[bool, "Whether the actual answer is correct"]


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

    invoke_kwargs = {
        "question": example_inputs["question"],
        "expected": example_outputs["answer"],
        "actual": run_outputs["answer"],
    }

    settings = get_settings()
    if settings.eval_jury_judges:
        return await minority_veto(
            key="correctness",
            prompt_name="sanctuary-eval-correctness",
            schema=CorrectnessGrade,
            invoke_kwargs=invoke_kwargs,
            score_field="correct",
            judges=settings.eval_jury_judges,
        )

    grader = await _get_grader(
        "correctness", "sanctuary-eval-correctness", CorrectnessGrade
    )
    grade: CorrectnessGrade = await grader.ainvoke(invoke_kwargs)

    return EvaluationResult(
        key="correctness",
        score=1 if grade["correct"] else 0,
        comment=grade["explanation"],
    )


class RelevanceGrade(TypedDict):
    """Structured output schema for the relevance grader.

    Attributes:
        explanation: Brief reasoning for the grade.
        relevant: Whether the answer is relevant to the question.
    """

    explanation: Annotated[str, "Brief reasoning for the grade"]
    relevant: Annotated[bool, "Whether the answer is relevant to the question"]


async def relevance(run: Run, example: Example | None) -> EvaluationResult:
    """LLM-as-judge evaluator that checks answer relevance.

    Determines whether the target function's answer is relevant to and
    addresses the original question, regardless of correctness.

    Args:
        run: The LangSmith run object containing the target function's outputs
            (must include an ``answer`` key).
        example: The LangSmith dataset example containing inputs
            (must include ``question``).

    Returns:
        An ``EvaluationResult`` with ``key`` set to ``relevance``, ``score``
        of 1 or 0, and a ``comment`` containing the grader's explanation.
    """
    run_outputs = _get_outputs(run, "outputs")
    example_inputs = _get_outputs(example, "inputs")

    missing = []
    if "question" not in example_inputs:
        missing.append("question (example.inputs)")
    if "answer" not in run_outputs:
        missing.append("answer (run.outputs)")

    if missing:
        return EvaluationResult(
            key="relevance",
            score=0,
            comment=f"Missing required keys: {', '.join(missing)}",
        )

    invoke_kwargs = {
        "question": example_inputs["question"],
        "answer": run_outputs["answer"],
    }

    settings = get_settings()
    if settings.eval_jury_judges:
        return await minority_veto(
            key="relevance",
            prompt_name="sanctuary-eval-relevance",
            schema=RelevanceGrade,
            invoke_kwargs=invoke_kwargs,
            score_field="relevant",
            judges=settings.eval_jury_judges,
        )

    grader = await _get_grader("relevance", "sanctuary-eval-relevance", RelevanceGrade)
    grade: RelevanceGrade = await grader.ainvoke(invoke_kwargs)

    return EvaluationResult(
        key="relevance",
        score=1 if grade["relevant"] else 0,
        comment=grade["explanation"],
    )


class GroundedGrade(TypedDict):
    """Structured output schema for the groundedness grader.

    Attributes:
        explanation: Brief reasoning for the grade.
        grounded: Whether the answer is grounded in the retrieved documents.
    """

    explanation: Annotated[str, "Brief reasoning for the grade"]
    grounded: Annotated[
        bool, "Whether the answer is grounded in the retrieved documents"
    ]


async def groundedness(run: Run, example: Example | None) -> EvaluationResult:
    """LLM-as-judge evaluator that checks answer groundedness.

    Determines whether the target function's answer is grounded in and
    supported by the retrieved documents, without hallucinated claims.

    Args:
        run: The LangSmith run object containing the target function's outputs
            (must include ``answer`` and ``documents`` keys).
        example: The LangSmith dataset example (unused but required by
            the evaluator signature).

    Returns:
        An ``EvaluationResult`` with ``key`` set to ``groundedness``, ``score``
        of 1 or 0, and a ``comment`` containing the grader's explanation.
    """
    run_outputs = _get_outputs(run, "outputs")

    missing = []
    if "answer" not in run_outputs:
        missing.append("answer (run.outputs)")
    if "documents" not in run_outputs:
        missing.append("documents (run.outputs)")

    if missing:
        return EvaluationResult(
            key="groundedness",
            score=0,
            comment=f"Missing required keys: {', '.join(missing)}",
        )

    invoke_kwargs = {
        "answer": run_outputs["answer"],
        "documents": _format_docs(run_outputs["documents"]),
    }

    settings = get_settings()
    if settings.eval_jury_judges:
        return await minority_veto(
            key="groundedness",
            prompt_name="sanctuary-eval-groundedness",
            schema=GroundedGrade,
            invoke_kwargs=invoke_kwargs,
            score_field="grounded",
            judges=settings.eval_jury_judges,
        )

    grader = await _get_grader(
        "groundedness", "sanctuary-eval-groundedness", GroundedGrade
    )
    grade: GroundedGrade = await grader.ainvoke(invoke_kwargs)

    return EvaluationResult(
        key="groundedness",
        score=1 if grade["grounded"] else 0,
        comment=grade["explanation"],
    )


class RetrievalRelevanceGrade(TypedDict):
    """Structured output schema for the retrieval relevance grader.

    Attributes:
        explanation: Brief reasoning for the grade.
        relevant: Whether the retrieved documents are relevant to the question.
    """

    explanation: Annotated[str, "Brief reasoning for the grade"]
    relevant: Annotated[
        bool, "Whether the retrieved documents are relevant to the question"
    ]


async def retrieval_relevance(run: Run, example: Example | None) -> EvaluationResult:
    """LLM-as-judge evaluator that checks retrieval relevance.

    Determines whether the retrieved documents are relevant to and
    contain information useful for answering the original question.

    Args:
        run: The LangSmith run object containing the target function's outputs
            (must include a ``documents`` key).
        example: The LangSmith dataset example containing inputs
            (must include ``question``).

    Returns:
        An ``EvaluationResult`` with ``key`` set to ``retrieval_relevance``,
        ``score`` of 1 or 0, and a ``comment`` containing the grader's
        explanation.
    """
    run_outputs = _get_outputs(run, "outputs")
    example_inputs = _get_outputs(example, "inputs")

    missing = []
    if "question" not in example_inputs:
        missing.append("question (example.inputs)")
    if "documents" not in run_outputs:
        missing.append("documents (run.outputs)")

    if missing:
        return EvaluationResult(
            key="retrieval_relevance",
            score=0,
            comment=f"Missing required keys: {', '.join(missing)}",
        )

    invoke_kwargs = {
        "question": example_inputs["question"],
        "documents": _format_docs(run_outputs["documents"]),
    }

    settings = get_settings()
    if settings.eval_jury_judges:
        return await minority_veto(
            key="retrieval_relevance",
            prompt_name="sanctuary-eval-retrieval-relevance",
            schema=RetrievalRelevanceGrade,
            invoke_kwargs=invoke_kwargs,
            score_field="relevant",
            judges=settings.eval_jury_judges,
        )

    grader = await _get_grader(
        "retrieval_relevance",
        "sanctuary-eval-retrieval-relevance",
        RetrievalRelevanceGrade,
    )
    grade: RetrievalRelevanceGrade = await grader.ainvoke(invoke_kwargs)

    return EvaluationResult(
        key="retrieval_relevance",
        score=1 if grade["relevant"] else 0,
        comment=grade["explanation"],
    )
