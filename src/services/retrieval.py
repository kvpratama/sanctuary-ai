import logging
import re
from collections.abc import AsyncGenerator

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_DIMENSIONS, get_settings
from src.db.client import get_supabase_client
from src.schemas.chat import (
    ChunksEvent,
    Citation,
    CitationsEvent,
    StreamEvent,
    TokenEvent,
)
from supabase import AsyncClient

logger = logging.getLogger(__name__)


async def retrieve_chunks(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
    *,
    client: AsyncClient | None = None,
) -> list[Document]:
    """Retrieve relevant chunks from Supabase vector store.

    Args:
        query: The user's question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve.
        client: Optional Supabase client to reuse for requests. When provided,
            this client is used instead of creating a new one via get_supabase_client.

    Returns:
        List of LangChain Document objects with page metadata.
    """
    # Generate embedding for the query
    settings = get_settings()
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.embedding_model}",
        google_api_key=settings.gemini_api_key,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )
    query_embedding = await embeddings_model.aembed_query(query)

    # Query Supabase for similar chunks with filters
    if client is None:
        client = await get_supabase_client()

    # Use RPC function for similarity search with filter parameter
    result = await client.rpc(
        "match_document_embeddings",
        {
            "query_embedding": query_embedding,
            "filter": {
                "document_id": document_id,
                "user_id": user_id,
            },
            "match_count": k,
        },
    ).execute()

    if not result.data:
        logger.debug("No retrieval results for document_id=%s", document_id)
        return []

    # Convert to LangChain Document format, filtering by similarity threshold
    chunks: list[Document] = []
    for row in result.data:  # ty: ignore[not-iterable]
        if row.get("similarity", 0) < settings.min_similarity:  # ty: ignore[unresolved-attribute]
            continue
        chunks.append(
            Document(
                page_content=row["content"],  # ty: ignore[invalid-argument-type, not-subscriptable]
                metadata={"page": row["metadata"].get("page")},  # ty: ignore[invalid-argument-type, not-subscriptable]
            )
        )

    return chunks


def extract_citations(answer: str, chunks: list[Document]) -> list[Citation]:
    """Parse page citations from the LLM answer and validate against retrieved chunks.

    Looks for ``[p. X]`` markers in the answer text and returns only those
    page numbers that also appear in the retrieved chunk metadata.

    Args:
        answer: The LLM-generated answer text containing ``[p. X]`` markers.
        chunks: List of retrieved Document objects with page metadata.

    Returns:
        Sorted list of Citation objects for pages both cited and present in chunks.
        Returns an empty list when no valid cited pages are found.
    """
    cited_pages = {int(m) for m in re.findall(r"\[p\.\s*(\d+)\]", answer)}
    available_pages = {
        page for chunk in chunks if (page := chunk.metadata.get("page")) is not None
    }
    valid_pages = sorted(p for p in cited_pages if p > 0 and p in available_pages)
    return [Citation(page=page) for page in valid_pages]


async def stream_answer_with_citations(
    query: str,
    chunks: list[Document],
) -> AsyncGenerator[StreamEvent, None]:
    """Stream LLM answer tokens, then yield a citations event.

    Yields:
        TokenEvent: Individual token strings as they arrive from the LLM.
        CitationsEvent: Final item containing the citations list.
    """
    if not chunks:
        yield TokenEvent(
            token="I don't have enough information to answer that question."
        )
        yield CitationsEvent(citations=[])
        return

    # Build context (same logic as generate_answer_with_citations)
    context_parts = []
    for chunk in chunks:
        page = chunk.metadata.get("page", "unknown")
        context_parts.append(f"[Page {page}]: {chunk.page_content}")

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions based ONLY on the "
                "provided document content. Do not use any outside knowledge. If the "
                "answer cannot be found in the content, say so.\n\n"
                "Always cite your sources using [p. X] format where X is the page number.",
            ),
            (
                "human",
                "Question: {query}\n\nContext:\n{context}",
            ),
        ]
    )

    settings = get_settings()
    llm = init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.llm_provider_base_url,
        temperature=0,
        streaming=True,
    )

    chain = prompt | llm

    full_answer = ""
    try:
        async for chunk in chain.astream({"query": query, "context": context}):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            token: str = content if isinstance(content, str) else str(content)
            if token:
                full_answer += token
                yield TokenEvent(token=token)
    except GeneratorExit:
        logger.warning(
            "Generator cancelled before completion, full_answer so far: %s",
            full_answer[:100],
        )
        return

    if not full_answer:
        yield TokenEvent(
            token="I don't have enough information to answer that question."
        )

    citations = extract_citations(full_answer, chunks)
    yield CitationsEvent(citations=citations)


async def stream_rag_pipeline(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
    *,
    client: AsyncClient | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Retrieve chunks then stream the LLM answer with citations.

    Eagerly calls ``retrieve_chunks``, then yields ``StreamEvent`` items
    from ``stream_answer_with_citations``.  This is the single entry point
    for the full RAG pipeline — used by both the chat router (streaming)
    and the eval target (drained to collect the final result).

    Args:
        query: The user's question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve.
        client: Optional Supabase client to reuse for requests.

    Yields:
        ChunksEvent with retrieved documents, then TokenEvent and CitationsEvent.
    """
    chunks = await retrieve_chunks(
        query=query,
        document_id=document_id,
        user_id=user_id,
        k=k,
        client=client,
    )

    yield ChunksEvent(
        chunks=[
            {"page_content": c.page_content, "page": c.metadata.get("page")}
            for c in chunks
        ]
    )

    async for event in stream_answer_with_citations(query=query, chunks=chunks):
        yield event
