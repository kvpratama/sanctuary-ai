"""Integration test for the eval pipeline.

Requires live Supabase, LLM, and LangSmith connections.
Skipped by default — run with ``uv run pytest -m integration``.
"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_eval_run_end_to_end() -> None:
    """Run the full evaluation pipeline against live services.

    This test calls ``main()`` which:
    1. Creates the LangSmith dataset if needed
    2. Runs ``langsmith.aevaluate()`` with the async RAG target
    3. Scores results with the correctness evaluator

    Requires a populated .env with valid SUPABASE_*, OPENAI_API_KEY,
    EVAL_LLM_*, and LANGSMITH_* credentials, plus real document/user
    IDs in ``src/eval/dataset.py``.
    """
    from src.eval.run import main

    # Should complete without raising
    await main()
