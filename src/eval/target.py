"""Async target function for LangSmith evaluation.

Wraps the RAG pipeline into the dict-in / dict-out shape that
``langsmith.evaluate()`` expects.
"""

from src.schemas.chat import ChunksEvent, CitationsEvent, RetrievedChunk, TokenEvent
from src.services.retrieval import stream_rag_pipeline


async def target(inputs: dict) -> dict:
    """Run the RAG pipeline on a single evaluation example.

    Args:
        inputs: Dictionary with ``question``, ``document_id``, and ``user_id``.

    Returns:
        Dictionary with ``answer`` (str), ``documents`` (list of dicts),
        and ``citations`` (list of dicts).
    """
    answer_parts: list[str] = []
    documents: list[RetrievedChunk] = []
    citation_pages: list[dict] = []

    async for event in stream_rag_pipeline(
        query=inputs["question"],
        document_id=inputs["document_id"],
        user_id=inputs["user_id"],
    ):
        if isinstance(event, ChunksEvent):
            documents = event.chunks
        elif isinstance(event, TokenEvent):
            answer_parts.append(event.token)
        elif isinstance(event, CitationsEvent):
            citation_pages = [{"page": c.page} for c in event.citations]

    return {
        "answer": "".join(answer_parts),
        "documents": documents,
        "citations": citation_pages,
    }
