import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from src.auth import AuthenticatedUser, get_authenticated_user
from src.schemas.chat import ChatRequest, ChunksEvent, CitationsEvent
from src.services.retrieval import stream_rag_pipeline

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
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> EventSourceResponse:
    """Stream an answer about a document using RAG via Server-Sent Events.

    Args:
        document_id: UUID of the document to query.
        request: Chat request containing the user's question.

    Returns:
        EventSourceResponse streaming token, citations, and done events.
    """

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        """Yield server-sent event dicts until the stream completes or the client disconnects.

        Each yielded dict contains ``event`` (event type) and ``data`` (JSON payload)
        keys conforming to the SSE protocol.

        Yields:
            A dict with ``event`` and ``data`` keys for each SSE frame.
        """
        try:
            async for event in stream_rag_pipeline(
                query=request.message,  # ty: ignore[invalid-argument-type]
                document_id=document_id,  # ty: ignore[invalid-argument-type]
                user_id=user.id,  # ty: ignore[invalid-argument-type]
                k=5,  # ty: ignore[invalid-argument-type]
                client=user.client,  # ty: ignore[invalid-argument-type]
            ):
                if isinstance(event, ChunksEvent):
                    continue
                elif isinstance(event, CitationsEvent):
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
