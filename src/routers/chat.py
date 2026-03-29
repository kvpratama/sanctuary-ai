import logging

from fastapi import APIRouter, HTTPException

from src.schemas.chat import ChatRequest, ChatResponse
from src.services.retrieval import generate_answer_with_citations, retrieve_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat/{document_id}", response_model=ChatResponse)
async def chat(
    document_id: str,
    request: ChatRequest,
) -> ChatResponse:
    """Answer a question about a document using RAG.

    Args:
        document_id: UUID of the document to query.
        request: Chat request containing the user's question.

    Returns:
        ChatResponse with answer and citations.

    Raises:
        HTTPException: If document not found or other errors.
    """
    # Hardcode user_id for now (as per requirements)
    user_id = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"

    try:
        # Retrieve relevant chunks
        chunks = await retrieve_chunks(
            query=request.message,
            document_id=document_id,
            user_id=user_id,
            k=5,
        )

        # Generate answer with citations
        answer, citations = await generate_answer_with_citations(
            query=request.message,
            chunks=chunks,
        )

        return ChatResponse(answer=answer, citations=citations)

    except Exception:
        logger.exception("Chat request failed for document %s", document_id)
        raise HTTPException(status_code=500, detail="Internal server error")
