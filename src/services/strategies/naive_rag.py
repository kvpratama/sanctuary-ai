"""Naive RAG strategy — direct retrieve then generate.

This is the original pipeline logic extracted from retrieval.py.
"""

from collections.abc import AsyncGenerator

from langsmith import traceable

from src.schemas.chat import ChunksEvent, RetrievedChunk, StreamEvent
from src.services.retrieval import retrieve_chunks, stream_answer_with_citations
from supabase import AsyncClient


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
    chunks = await retrieve_chunks(
        query=query,  # ty: ignore[invalid-argument-type]
        document_id=document_id,  # ty: ignore[invalid-argument-type]
        user_id=user_id,  # ty: ignore[invalid-argument-type]
        k=k,  # ty: ignore[invalid-argument-type]
        client=client,  # ty: ignore[invalid-argument-type]
    )

    yield ChunksEvent(
        chunks=[
            RetrievedChunk(page_content=c.page_content, page=c.metadata.get("page"))
            for c in chunks
        ]
    )

    async for event in stream_answer_with_citations(query=query, chunks=chunks):  # ty: ignore[invalid-argument-type]
        yield event
