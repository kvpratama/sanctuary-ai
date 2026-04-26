"""Step-Back RAG strategy — abstract the query before retrieval.

Generates a broader, more abstract version of the user's question,
retrieves chunks for both the original and step-back queries, fuses
results using Reciprocal Rank Fusion (RRF), and generates an answer
from the merged context.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

from langchain.chat_models import init_chat_model
from langsmith import traceable

from src.config import get_settings
from src.prompts.manager import pull_eval_prompt
from src.schemas.chat import ChunksEvent, RetrievedChunk, StreamEvent
from src.services.strategies.core import (
    fuse_rrf,
    retrieve_chunks,
    stream_answer_with_citations,
)
from supabase import AsyncClient

logger = logging.getLogger(__name__)


@traceable(metadata={"rag_strategy": "step_back"})
async def generate_step_back_query(query: str) -> str:
    """Generate a broader step-back question using an LLM.

    Args:
        query: The original user question.

    Returns:
        The step-back question string. Falls back to the original if the LLM
        returns an empty response.
    """
    try:
        settings = get_settings()
        llm = init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.llm_provider_base_url,
            temperature=0,
        )

        prompt = await pull_eval_prompt("sanctuary-step-back")
        chain = prompt | llm
        response = await chain.ainvoke({"query": query})

        content = response.content if hasattr(response, "content") else ""
        step_back = content.strip() if isinstance(content, str) else ""
        if not step_back:
            logger.warning(
                "Step-back query generation returned empty, falling back to original query"
            )
            return query

        logger.info(
            "Step-back query generated, original_len=%d, step_back_len=%d",
            len(query),
            len(step_back),
        )
        logger.debug("Step-back query: '%.64s' -> '%.64s'", query, step_back)
        return step_back
    except Exception:
        logger.warning(
            "Step-back query generation failed, falling back to original query",
            exc_info=True,
        )
        return query


@traceable(metadata={"rag_strategy": "step_back"})
async def execute(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
    *,
    client: AsyncClient | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Run the step-back RAG strategy.

    Generates a broader step-back query, retrieves chunks for both the
    original and step-back queries, fuses results using RRF, and
    generates an answer using the original query.

    Args:
        query: The user's original question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve.
        client: Optional Supabase client to reuse.

    Yields:
        ChunksEvent with RRF-fused documents, then TokenEvent and CitationsEvent.
    """
    logger.info(
        "step_back: generating step-back query document_id=%s, k=%d, query_len=%d",
        document_id,
        k,
        len(query),
    )
    step_back = await generate_step_back_query(query=query)  # ty: ignore[invalid-argument-type]

    logger.debug("step_back: step-back query='%.64s'", step_back)

    if step_back == query:
        logger.info(
            "step_back: step-back query identical to original, short-circuiting duplicate retrieval"
        )
        fused = await retrieve_chunks(
            query=query,  # ty: ignore[invalid-argument-type]
            document_id=document_id,  # ty: ignore[invalid-argument-type]
            user_id=user_id,  # ty: ignore[invalid-argument-type]
            k=k,  # ty: ignore[invalid-argument-type]
            client=client,  # ty: ignore[invalid-argument-type]
        )
    else:
        logger.info("step_back: retrieving chunks for original and step-back queries")
        original_chunks, step_back_chunks = await asyncio.gather(
            retrieve_chunks(
                query=query,  # ty: ignore[invalid-argument-type]
                document_id=document_id,  # ty: ignore[invalid-argument-type]
                user_id=user_id,  # ty: ignore[invalid-argument-type]
                k=k,  # ty: ignore[invalid-argument-type]
                client=client,  # ty: ignore[invalid-argument-type]
            ),
            retrieve_chunks(
                query=step_back,  # ty: ignore[invalid-argument-type]
                document_id=document_id,  # ty: ignore[invalid-argument-type]
                user_id=user_id,  # ty: ignore[invalid-argument-type]
                k=k,  # ty: ignore[invalid-argument-type]
                client=client,  # ty: ignore[invalid-argument-type]
            ),
        )
        fused = fuse_rrf([original_chunks, step_back_chunks], max_chunks=k)

    logger.info("step_back: using %d fused chunks", len(fused))
    yield ChunksEvent(
        chunks=[
            RetrievedChunk(page_content=c.page_content, page=c.metadata.get("page"))
            for c in fused
        ]
    )

    logger.info("step_back: streaming answer with original query")
    async for event in stream_answer_with_citations(query=query, chunks=fused):  # ty: ignore[invalid-argument-type]
        yield event
