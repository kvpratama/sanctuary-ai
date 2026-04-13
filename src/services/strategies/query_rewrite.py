"""Query Rewrite RAG strategy — rewrite query before retrieval.

Uses an LLM to rewrite the user's query into a better search query
optimized for semantic similarity, then runs the standard retrieve → generate flow.
"""

import logging
from collections.abc import AsyncGenerator

from langchain.chat_models import init_chat_model
from langsmith import traceable

from src.config import get_settings
from src.prompts.manager import pull_eval_prompt
from src.schemas.chat import ChunksEvent, RetrievedChunk, StreamEvent
from src.services.strategies.core import retrieve_chunks, stream_answer_with_citations
from supabase import AsyncClient

logger = logging.getLogger(__name__)


@traceable(metadata={"rag_strategy": "query_rewrite"})
async def rewrite_query(query: str) -> str:
    """Rewrite a user query into a search-optimized query using an LLM.

    Args:
        query: The original user question.

    Returns:
        The rewritten query string. Falls back to the original if the LLM
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

        prompt = await pull_eval_prompt("sanctuary-query-rewrite")
        chain = prompt | llm
        response = await chain.ainvoke({"query": query})

        content = response.content if hasattr(response, "content") else ""
        rewritten = content.strip() if isinstance(content, str) else ""
        if not rewritten:
            logger.warning(
                "Query rewrite returned empty, falling back to original query"
            )
            return query

        logger.info(
            "Query rewritten, original_len=%d, rewritten_len=%d",
            len(query),
            len(rewritten),
        )
        logger.debug("Query rewritten: '%.64s' -> '%.64s'", query, rewritten)
        return rewritten
    except Exception:
        logger.warning("Query rewrite failed, falling back to original query")
        return query


@traceable(metadata={"rag_strategy": "query_rewrite"})
async def execute(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
    *,
    client: AsyncClient | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Run the query rewrite RAG strategy.

    Rewrites the query for better retrieval, then retrieves and generates
    using the original query for answer generation.

    Args:
        query: The user's original question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve.
        client: Optional Supabase client to reuse.

    Yields:
        ChunksEvent with retrieved documents, then TokenEvent and CitationsEvent.
    """
    logger.info(
        "query_rewrite: rewriting query document_id=%s, k=%d, query_len=%d",
        document_id,
        k,
        len(query),
    )
    rewritten = await rewrite_query(query=query)  # ty: ignore[invalid-argument-type]

    logger.info(
        "query_rewrite: retrieving chunks with rewritten query, query_len=%d",
        len(rewritten),
    )
    logger.debug("query_rewrite: rewritten query='%.64s'", rewritten)
    chunks = await retrieve_chunks(
        query=rewritten,  # ty: ignore[invalid-argument-type]
        document_id=document_id,  # ty: ignore[invalid-argument-type]
        user_id=user_id,  # ty: ignore[invalid-argument-type]
        k=k,  # ty: ignore[invalid-argument-type]
        client=client,  # ty: ignore[invalid-argument-type]
    )

    logger.info("query_rewrite: retrieved %d chunks", len(chunks))
    yield ChunksEvent(
        chunks=[
            RetrievedChunk(page_content=c.page_content, page=c.metadata.get("page"))
            for c in chunks
        ]
    )

    logger.info("query_rewrite: streaming answer with original query")
    async for event in stream_answer_with_citations(query=query, chunks=chunks):  # ty: ignore[invalid-argument-type]
        yield event
