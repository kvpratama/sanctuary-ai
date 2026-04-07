"""LangSmith dynamic prompt management with fallback handling."""

import asyncio
import logging
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.prompt_cache import configure_global_prompt_cache
from langsmith.utils import LangSmithNotFoundError

logger = logging.getLogger(__name__)

# Configure local global cache API calls to refresh every 5 minutes
configure_global_prompt_cache(ttl_seconds=300)

CORRECTNESS_PROMPT_HARDCODED = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grading assistant. Grade whether the ACTUAL answer "
            "is correct given the EXPECTED answer. You must first respond with a brief "
            "explanation of your reasoning, and then finally output whether the answer is correct.",
        ),
        (
            "human",
            "Question: {question}\nExpected: {expected}\nActual: {actual}",
        ),
    ]
)

RELEVANCE_PROMPT_HARDCODED = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grading assistant. Determine whether the ANSWER is relevant to "
            "the QUESTION. The answer does not need to be correct, only on-topic and "
            "responsive to what was asked. You must first respond with a brief explanation "
            "of your reasoning, and then finally output whether the answer is relevant.",
        ),
        (
            "human",
            "Question: {question}\nAnswer: {answer}",
        ),
    ]
)

GROUNDEDNESS_PROMPT_HARDCODED = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grading assistant. Determine whether the ANSWER is grounded in "
            "and supported by the provided DOCUMENTS. The answer should not contain "
            "claims that cannot be traced back to the documents. You must first respond "
            "with a brief explanation of your reasoning, and then finally output whether "
            "the answer is grounded.",
        ),
        (
            "human",
            "Answer: {answer}\nDocuments: {documents}",
        ),
    ]
)

RAG_PROMPT_HARDCODED = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based ONLY on the "
            "provided document content. Do not use any outside knowledge. If the "
            "answer cannot be found in the content, say so.\n\n"
            "Always cite your sources using [p. X] format where X is the page number.",
        ),
        (
            "human",
            "Question: {query}\n\nContext:\n{context}",
        ),
    ]
)

FALLBACK_PROMPTS: dict[str, ChatPromptTemplate] = {
    "sanctuary-eval-correctness": CORRECTNESS_PROMPT_HARDCODED,
    "sanctuary-eval-relevance": RELEVANCE_PROMPT_HARDCODED,
    "sanctuary-eval-groundedness": GROUNDEDNESS_PROMPT_HARDCODED,
    "sanctuary-rag-prompt": RAG_PROMPT_HARDCODED,
}


@lru_cache(maxsize=1)
def _get_client() -> Client:
    """Return a cached LangSmith client instance.

    Returns:
        The initialized Client.
    """
    return Client()


async def pull_eval_prompt(name: str) -> ChatPromptTemplate:
    """Pull a prompt by name from LangSmith hub, falling back to a hardcoded version.

    This function leverages the global prompt cache configured with a 5-minute TTL.
    If pulling fails due to network issues or missing prompts, it gracefully falls
    back to the statically defined ChatPromptTemplate and logs a warning.

    Args:
        name: Name of the prompt to pull (e.g. ``sanctuary-eval-correctness``).

    Returns:
        The pulled or fallback ChatPromptTemplate.

    Raises:
        ValueError: If the pull fails and no fallback is registered for this name.
    """
    fallback = FALLBACK_PROMPTS.get(name)
    try:
        client = _get_client()
        prompt = await asyncio.to_thread(client.pull_prompt, name)
        return prompt
    except (LangSmithNotFoundError, Exception) as e:
        logger.warning(
            "Failed to pull prompt %s from LangSmith: %s. Using fallback.", name, e
        )
        if fallback:
            return fallback
        raise ValueError(f"Prompt {name} pull failed and no fallback exists.") from e


def push_eval_prompts() -> None:
    """Seed fallback prompts to the LangSmith prompt hub (skip if already exists).

    This intentionally does NOT overwrite existing prompts so that prompt
    iteration can happen via the LangSmith UI without requiring code changes
    or a git push.  The hardcoded ``FALLBACK_PROMPTS`` serve only as initial
    seeds and as offline fallbacks when the hub is unreachable.
    """
    logger.info("Initializing LangSmith prompt push...")
    client = _get_client()
    for name, prompt in FALLBACK_PROMPTS.items():
        try:
            client.pull_prompt(name)
            logger.info("Prompt '%s' already exists. Skipping push.", name)
            print(f"Prompt '{name}' already exists. Skipping push.")
            continue
        except LangSmithNotFoundError:
            url = client.push_prompt(name, object=prompt)
            logger.info("Successfully pushed '%s': %s", name, url)
            print(f"Successfully pushed '{name}': {url}")
        except Exception as e:
            logger.error("Error pushing '%s': %s", name, e)
            print(f"Error pushing '{name}': {e}")
