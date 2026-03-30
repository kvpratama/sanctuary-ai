import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from src.schemas.chat import ChatRequest, CitationsEvent
from src.services.retrieval import retrieve_chunks, stream_answer_with_citations

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/chat/{document_id}",
    response_class=EventSourceResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat(
    document_id: str,
    request: ChatRequest,
) -> EventSourceResponse:
    """Stream an answer about a document using RAG via Server-Sent Events.

    Args:
        document_id: UUID of the document to query.
        request: Chat request containing the user's question.

    Returns:
        EventSourceResponse streaming token, citations, and done events.

    Raises:
        HTTPException: If document not found or retrieval fails.
    """
    user_id = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"

    try:
        chunks = await retrieve_chunks(
            query=request.message,
            document_id=document_id,
            user_id=user_id,
            k=5,
        )
    except Exception as e:
        logger.exception("Chat request failed for document %s", document_id)
        raise HTTPException(status_code=500, detail=str(e)) from e

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        """Yield server-sent event dicts until the stream completes or the client disconnects.

        Each yielded dict contains ``event`` (event type) and ``data`` (JSON payload)
        keys conforming to the SSE protocol.

        Yields:
            A dict with ``event`` and ``data`` keys for each SSE frame.
        """
        try:
            async for event in stream_answer_with_citations(
                query=request.message,
                chunks=chunks,
            ):
                if isinstance(event, CitationsEvent):
                    yield {
                        "event": "citations",
                        "data": json.dumps([c.model_dump() for c in event.citations]),
                    }
                    yield {
                        "event": "done",
                        "data": json.dumps({}),
                    }
                else:
                    yield {
                        "event": "token",
                        "data": json.dumps(event.token),
                    }
        except Exception:
            logger.exception("Streaming failed for document %s", document_id)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"detail": "An error occurred while generating the answer."}
                ),
            }
            yield {
                "event": "done",
                "data": json.dumps({}),
            }

    return EventSourceResponse(event_generator())
