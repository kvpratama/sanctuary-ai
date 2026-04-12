"""CLI entry point for running RAG correctness evaluation.

Usage::

    uv run python -m src.eval.run
"""

import asyncio

from dotenv import load_dotenv
from langsmith import Client, aevaluate

from src.config import get_settings
from src.eval.dataset import ensure_dataset
from src.eval.evaluators import (
    correctness,
    groundedness,
    relevance,
    retrieval_relevance,
)
from src.eval.target import target


async def main() -> None:
    """Run end-to-end correctness evaluation against LangSmith.

    Creates the dataset if it does not exist, then runs
    ``langsmith.aevaluate()`` with the async RAG target and correctness
    evaluator.
    """
    load_dotenv()

    settings = get_settings()
    if settings.eval_jury_judges:
        judge_names = [j.model for j in settings.eval_jury_judges]
        print(f"Jury mode: {len(judge_names)} judges — {', '.join(judge_names)}")
    else:
        print("Single-judge mode (no EVAL_JURY_JUDGES configured)")

    client = Client()
    dataset_name = ensure_dataset(client)

    results = await aevaluate(
        target,
        data=dataset_name,
        evaluators=[correctness, relevance, groundedness, retrieval_relevance],
        experiment_prefix=f"sanctuary_{settings.rag_strategy}",
        max_concurrency=1,
    )


if __name__ == "__main__":
    asyncio.run(main())
