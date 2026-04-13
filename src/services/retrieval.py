import logging
from collections.abc import AsyncGenerator

from langsmith import traceable

from src.config import get_settings
from src.schemas.chat import StreamEvent
from src.services.strategies.registry import get_strategy
from supabase import AsyncClient

logger = logging.getLogger(__name__)


@traceable
async def stream_rag_pipeline(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
    *,
    client: AsyncClient | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Dispatch to the configured RAG strategy and stream results.

    Reads ``settings.rag_strategy`` to select the pipeline variant.
    Each strategy must yield ``ChunksEvent``, then ``TokenEvent``/``CitationsEvent``.

    Args:
        query: The user's question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve.
        client: Optional Supabase client to reuse.

    Yields:
        ChunksEvent with retrieved documents, then TokenEvent and CitationsEvent.
    """
    settings = get_settings()
    try:
        strategy = get_strategy(settings.rag_strategy)
    except ValueError:
        logger.warning(
            "Unknown RAG strategy '%s', falling back to 'naive_rag'",
            settings.rag_strategy,
        )
        strategy = get_strategy("naive_rag")

    async for event in strategy(
        query=query,
        document_id=document_id,
        user_id=user_id,
        k=k,
        client=client,
    ):
        yield event
