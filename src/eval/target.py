"""Async target function for LangSmith evaluation.

Wraps the RAG pipeline (retrieve_chunks → stream_answer_with_citations)
into the dict-in / dict-out shape that ``langsmith.evaluate()`` expects.
"""

from langchain_core.documents import Document

from src.schemas.chat import CitationsEvent, TokenEvent
from src.services.retrieval import retrieve_chunks, stream_answer_with_citations


async def target(inputs: dict) -> dict:
    """Run the RAG pipeline on a single evaluation example.

    Args:
        inputs: Dictionary with ``question``, ``document_id``, and ``user_id``.

    Returns:
        Dictionary with ``answer`` (str), ``documents`` (list of dicts),
        and ``citations`` (list of dicts).
    """
    question: str = inputs["question"]
    document_id: str = inputs["document_id"]
    user_id: str = inputs["user_id"]

    # Retrieve using the service-role client (no auth / RLS bypass)
    chunks: list[Document] = await retrieve_chunks(
        query=question,
        document_id=document_id,
        user_id=user_id,
    )

    # Drain the streaming generator to collect the full answer + citations
    answer_parts: list[str] = []
    citation_pages: list[dict] = []

    async for event in stream_answer_with_citations(
        query=question,
        chunks=chunks,
    ):
        if isinstance(event, TokenEvent):
            answer_parts.append(event.token)
        elif isinstance(event, CitationsEvent):
            citation_pages = [{"page": c.page} for c in event.citations]

    # Serialise chunks into JSON-friendly dicts
    documents = [
        {"page_content": chunk.page_content, "page": chunk.metadata.get("page")}
        for chunk in chunks
    ]

    return {
        "answer": "".join(answer_parts),
        "documents": documents,
        "citations": citation_pages,
    }
