"""Naive RAG strategy — direct retrieve then generate.

This is the original pipeline logic extracted from retrieval.py.
"""

import logging
from collections.abc import AsyncGenerator

from langsmith import traceable

from src.schemas.chat import ChunksEvent, RetrievedChunk, StreamEvent
from src.services.strategies.core import retrieve_chunks, stream_answer_with_citations
from supabase import AsyncClient

logger = logging.getLogger(__name__)


@traceable(metadata={"rag_strategy": "naive_rag"})
async def execute(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
    *,
    client: AsyncClient | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Run the naive_rag RAG pipeline: retrieve chunks then stream answer.

    Args:
        query: The user's question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve.
        client: Optional Supabase client to reuse.

    Yields:
        ChunksEvent with retrieved documents, then TokenEvent and CitationsEvent.
    """
    logger.info(
        "naive_rag: retrieving chunks document_id=%s, k=%d, query_len=%d",
        document_id,
        k,
        len(query),
    )
    logger.debug("naive_rag: query='%.64s'", query)
    chunks = await retrieve_chunks(
        query=query,  # ty: ignore[invalid-argument-type]
        document_id=document_id,  # ty: ignore[invalid-argument-type]
        user_id=user_id,  # ty: ignore[invalid-argument-type]
        k=k,  # ty: ignore[invalid-argument-type]
        client=client,  # ty: ignore[invalid-argument-type]
    )

    logger.info("naive_rag: retrieved %d chunks", len(chunks))
    yield ChunksEvent(
        chunks=[
            RetrievedChunk(page_content=c.page_content, page=c.metadata.get("page"))
            for c in chunks
        ]
    )

    logger.info("naive_rag: streaming answer")
    async for event in stream_answer_with_citations(query=query, chunks=chunks):  # ty: ignore[invalid-argument-type]
        yield event
