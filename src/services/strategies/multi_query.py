"""Multi-Query RAG strategy — generate query variants and merge results.

Generates multiple query variants using an LLM, retrieves chunks for each,
deduplicates, and generates an answer from the merged context.
"""

import logging
from collections.abc import AsyncGenerator
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langsmith import traceable
from pydantic import BaseModel, Field

from src.config import get_settings
from src.prompts.manager import pull_eval_prompt
from src.schemas.chat import ChunksEvent, RetrievedChunk, StreamEvent
from src.services.retrieval import retrieve_chunks, stream_answer_with_citations
from supabase import AsyncClient

logger = logging.getLogger(__name__)


class QueryVariants(BaseModel):
    """A list of query variants generated from the original question."""

    variants: list[str] = Field(
        description="List of rephrased query variants for retrieval."
    )


@traceable(metadata={"rag_strategy": "multi_query"})
async def generate_query_variants(query: str, *, n: int = 3) -> list[str]:
    """Generate multiple query variants using an LLM.

    Args:
        query: The original user question.
        n: Number of variants to generate.

    Returns:
        A list of query variant strings. Falls back to [query] if the LLM
        returns an empty or unparseable response.
    """
    try:
        settings = get_settings()
        llm = init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.llm_provider_base_url,
            temperature=0.7,
        )

        structured_llm = llm.with_structured_output(QueryVariants)
        prompt = await pull_eval_prompt("sanctuary-multi-query")
        chain = prompt | structured_llm
        result = await chain.ainvoke({"query": query, "n": n})
        response = cast(QueryVariants, result)

        if not response.variants:
            logger.warning(
                "Multi-query generation returned empty, falling back to original"
            )
            return [query]

        logger.info(
            "Generated %d query variants from: '%s'", len(response.variants), query
        )
        return response.variants
    except Exception:
        logger.warning("Multi-query generation failed, falling back to original query")
        return [query]


@traceable(metadata={"rag_strategy": "multi_query"})
async def execute(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
    *,
    client: AsyncClient | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Run the multi-query RAG strategy.

    Generates query variants, retrieves chunks for each, deduplicates,
    and generates an answer using the original query.

    Args:
        query: The user's original question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve per variant.
        client: Optional Supabase client to reuse.

    Yields:
        ChunksEvent with deduplicated documents, then TokenEvent and CitationsEvent.
    """
    variants = await generate_query_variants(query, n=3)  # ty: ignore[invalid-argument-type]

    all_chunks: list[Document] = []
    for variant in variants:
        chunks = await retrieve_chunks(
            query=variant,  # ty: ignore[invalid-argument-type]
            document_id=document_id,  # ty: ignore[invalid-argument-type]
            user_id=user_id,  # ty: ignore[invalid-argument-type]
            k=k,  # ty: ignore[invalid-argument-type]
            client=client,  # ty: ignore[invalid-argument-type]
        )
        all_chunks.extend(chunks)

    unique_chunks = deduplicate_chunks(all_chunks)

    yield ChunksEvent(
        chunks=[
            RetrievedChunk(page_content=c.page_content, page=c.metadata.get("page"))
            for c in unique_chunks
        ]
    )

    async for event in stream_answer_with_citations(query=query, chunks=unique_chunks):  # ty: ignore[invalid-argument-type]
        yield event


def deduplicate_chunks(chunks: list[Document]) -> list[Document]:
    """Remove duplicate chunks by page_content, preserving first-seen order.

    Args:
        chunks: List of Document objects, potentially with duplicates.

    Returns:
        Deduplicated list preserving insertion order.
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique.append(chunk)
    return unique
