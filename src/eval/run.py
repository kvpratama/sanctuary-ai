"""CLI entry point for running RAG correctness evaluation.

Usage::

    uv run python -m src.eval.run
"""

import asyncio

from dotenv import load_dotenv
from langsmith import Client, aevaluate

from src.eval.dataset import ensure_dataset
from src.eval.evaluators import correctness
from src.eval.target import target


async def main() -> None:
    """Run end-to-end correctness evaluation against LangSmith.

    Creates the dataset if it does not exist, then runs
    ``langsmith.aevaluate()`` with the async RAG target and correctness
    evaluator.
    """
    load_dotenv()

    client = Client()
    dataset_name = ensure_dataset(client)

    results = await aevaluate(
        target,
        data=dataset_name,
        evaluators=[correctness],
        experiment_prefix="sanctuary",
    )

    print(results)


if __name__ == "__main__":
    asyncio.run(main())
